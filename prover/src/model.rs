use ark_std::{log2, rand::RngCore};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::MultilinearExtension;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};
use hcs_common::{
    InferenceProof, NIOTensor, NodeCommitment, NodeCommitmentState, Poly, SmallNIO, Tensor
};

use crate::NodeOpsProve;

pub struct ProvableModel<F, S, PCS, ST>
where
    F: PrimeField + Absorb + From<ST> + From<ST::LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
{
    pub nodes: Vec<Box<dyn NodeOpsProve<F, S, PCS, ST>>>,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
}

impl<F, S, PCS, ST> ProvableModel<F, S, PCS, ST>
where
    F: PrimeField + Absorb + From<ST> + From<ST::LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: SmallNIO,
{
    pub fn setup_keys<R>(
        &self,
        rng: &mut R,
    ) -> Result<(PCS::CommitterKey, PCS::VerifierKey), PCS::Error>
    where
        F: PrimeField + Absorb + From<ST> + From<ST::LT>,
        S: CryptographicSponge,
        PCS: PolynomialCommitment<F, Poly<F>, S>,
        R: RngCore,
    {
        let num_vars = self.nodes.iter().map(|n| n.com_num_vars()).max().unwrap();

        let pp = PCS::setup(1, Some(num_vars), rng).unwrap();

        // Make sure supported_degree, supported_hiding_bound and
        // enforced_degree_bounds have a consistent meaning across ML PCSs and
        // we are using them securely.
        PCS::trim(&pp, 0, 0, None)
    }

    /// Unlike the node's `padded_evaluate`, the model's `padded_evaluate` accepts unpadded input
    /// and first re-sizes it before running inference.
    pub fn padded_evaluate(&self, input: Tensor<ST>) -> Tensor<ST> {
        // TODO sanity check: input shape matches model input shape

        let input = input.compact_resize(
            // TODO this functionality is so common we might as well make it an #[inline] function
            self.input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            ST::ZERO,
        );

        // TODO switch to reference in reshape?
        self.nodes
            .iter()
            .fold(NIOTensor::S(input), |output, node| {
                node.padded_evaluate(&output)
            })
            .unwrap_small()
            .compact_resize(self.output_shape.clone(), ST::ZERO)
    }

    pub fn prove_inference(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
        sponge: &mut S,
        node_coms: &Vec<NodeCommitment<F, S, PCS>>,
        node_com_states: &Vec<NodeCommitmentState<F, S, PCS>>,
        input: Tensor<ST>,
    ) -> InferenceProof<F, S, PCS, ST> {
        // TODO Absorb public parameters into s (to be determined what exactly)

        let output = input.compact_resize(
            self.input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            ST::ZERO,
        );

        let output_f: Vec<F> = output.values().iter().map(|x| F::from(*x)).collect();

        let mut output = NIOTensor::S(output);

        // First pass: computing node values
        // TODO handling F and QSmallType is inelegant; we might want to switch
        // to F for IO in NodeOps::prove
        let mut node_outputs = vec![output.clone()];
        let mut node_output_mles = vec![Poly::from_evaluations_vec(
            log2(output_f.len()) as usize,
            output_f,
        )];

        for node in &self.nodes {
            output = node.padded_evaluate(&output);

            let output_f: Vec<F> = match &output {
                NIOTensor::S(o) => o.values().iter().map(|x| F::from(*x)).collect(),
                NIOTensor::L(o) => o.values().iter().map(|x| F::from(*x)).collect(),
            };

            node_outputs.push(output.clone());
            node_output_mles.push(Poly::from_evaluations_vec(
                log2(output_f.len()) as usize,
                output_f,
            ));
        }

        let input_node_f = node_output_mles.first().unwrap().to_evaluations();
        let output_node_f = node_output_mles.last().unwrap().to_evaluations();

        // Committing to node outputs as MLEs (individual per node for now)
        let labeled_output_mles: Vec<LabeledPolynomial<F, Poly<F>>> = node_output_mles
            .into_iter()
            .map(|mle|
            // TODO change dummy label once we e.g. have given numbers to the
            // nodes in the model: fc_1, fc_2, relu_1, etc.
            // TODO maybe we don't need to clone, if `prove` can take a reference
            LabeledPolynomial::new(
                "dummy".to_string(),
                mle,
                None,
                None,
            ))
            .collect();

        let (output_coms, output_com_states) = PCS::commit(ck, &labeled_output_mles, rng).unwrap();

        // Absorb all commitments into the sponge
        sponge.absorb(&output_coms);

        // TODO Prove that all commited NIOs live in the right range (to be
        // discussed)

        let mut node_proofs = Vec::new();

        // Second pass: proving
        for (((((node, node_com), node_com_state), values), l_v_coms), v_coms_states) in self
            .nodes
            .iter()
            .zip(node_coms.iter())
            .zip(node_com_states.iter())
            .zip(labeled_output_mles.windows(2))
            .zip(output_coms.windows(2))
            .zip(output_com_states.windows(2))
        {
            node_proofs.push(node.prove(
                ck,
                sponge,
                node_com,
                node_com_state,
                &values[0],
                &l_v_coms[0],
                &v_coms_states[0],
                &values[1],
                &l_v_coms[1],
                &v_coms_states[1],
            ));
        }

        // Opening model IO
        // TODO maybe this can be made more efficient by not committing to the
        // output nodes and instead working witht their plain values all along,
        // but that would require messy node-by-node handling
        let mut node_outputs = node_outputs.into_iter();
        let input_node = node_outputs.next().unwrap();
        let output_node = node_outputs.last().unwrap();

        let input_labeled_value = labeled_output_mles.first().unwrap();
        let input_node_com = output_coms.first().unwrap();
        let input_node_com_state = output_com_states.first().unwrap();

        let output_labeled_value = labeled_output_mles.last().unwrap();
        let output_node_com = output_coms.last().unwrap();
        let output_node_com_state = output_com_states.last().unwrap();

        // Absorb the model IO output and squeeze the challenge point
        // Absorb the plain output and squeeze the challenge point
        sponge.absorb(&input_node_f);
        sponge.absorb(&output_node_f);
        let input_challenge_point =
            sponge.squeeze_field_elements(log2(input_node_f.len()) as usize);
        let output_challenge_point =
            sponge.squeeze_field_elements(log2(output_node_f.len()) as usize);

        // TODO we have to pass rng, not None, but it has been moved before
        // fix this once we have decided how to handle the cumbersome
        // Option<&mut rng...>
        let input_opening_proof = PCS::open(
            ck,
            [input_labeled_value],
            [input_node_com],
            &input_challenge_point,
            sponge,
            [input_node_com_state],
            None,
        )
        .unwrap();

        // TODO we have to pass rng, not None, but it has been moved before
        // fix this once we have decided how to handle the cumbersome
        // Option<&mut rng...>
        let output_opening_proof = PCS::open(
            ck,
            [output_labeled_value],
            [output_node_com],
            &output_challenge_point,
            sponge,
            [output_node_com_state],
            None,
        )
        .unwrap();

        // TODO prove that inputs match input commitments?
        InferenceProof {
            inputs: vec![input_node],
            outputs: vec![output_node],
            node_value_commitments: output_coms,
            node_proofs,
            input_opening_proofs: vec![input_opening_proof],
            output_opening_proofs: vec![output_opening_proof],
        }
    }

    pub fn commit(
        &self,
        ck: &PCS::CommitterKey,
        _rng: Option<&mut dyn RngCore>,
    ) -> Vec<(NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>)> {
        // TODO blindly passing None, likely need to change to get hiding
        self.nodes.iter().map(|n| n.commit(ck, None)).collect()
    }
}
