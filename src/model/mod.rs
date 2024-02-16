use ark_std::{log2, rand::RngCore};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};

use crate::hidden_model::HiddenModel;
use crate::model::nodes::{NodeOps, NodeOpsSNARK};
use crate::proofs::InferenceProof;
use crate::qarray::QArray;
use crate::Poly;
use crate::{model::nodes::Node, quantization::QSmallType};

use self::nodes::{NodeCommitment, NodeCommitmentState};

mod examples;
pub(crate) mod nodes;
mod reshaping;

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all nodes to use the same PCS; this might change
// in the future
pub(crate) struct Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    input_shape: Vec<usize>,
    output_shape: Vec<usize>,
    nodes: Vec<Node<F, S, PCS>>,
}

impl<F, S, PCS> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_shape: Vec<usize>, nodes: Vec<Node<F, S, PCS>>) -> Self {
        // An empty model would cause panics later down the line e.g. when
        // determining the number of variables needed to commit to it.
        assert!(!nodes.is_empty(), "A model cannot have no nodes",);

        Self {
            input_shape,
            output_shape: nodes.last().unwrap().shape(),
            nodes,
        }
    }

    pub(crate) fn setup_keys<R: RngCore>(
        &self,
        rng: &mut R,
    ) -> Result<(PCS::CommitterKey, PCS::VerifierKey), PCS::Error> {
        let num_vars = self.nodes.iter().map(|n| n.com_num_vars()).max().unwrap();

        let pp = PCS::setup(1, Some(num_vars), rng).unwrap();

        // Make sure supported_degree, supported_hiding_bound and
        // enforced_degree_bounds have a consistent meaning across ML PCSs and
        // we are using them securely.
        PCS::trim(&pp, 0, 0, None)
    }

    pub(crate) fn evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        let mut output = input;
        for node in &self.nodes {
            output = node.evaluate(output);
        }
        output
    }

    pub(crate) fn padded_evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // TODO sanity check: input shape matches model input shape

        let mut output = input.compact_resize(
            // TODO this functionality is so common we might as well make it an #[inline] function
            self.input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            0,
        );

        for node in &self.nodes {
            output = node.padded_evaluate(output);
        }

        // TODO switch to reference in reshape?
        output.compact_resize(self.output_shape.clone(), 0)
    }

    pub(crate) fn prove_inference(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
        s: &mut S,
        node_commitments: Vec<NodeCommitment<F, S, PCS>>,
        input: QArray<QSmallType>,
    ) -> InferenceProof<F, S, PCS> {
        // TODO Absorb public parameters into s (to be determined what exactly)

        let mut output = input.compact_resize(
            self.input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            0,
        );

        let output_f = output.values().iter().map(|x| F::from(*x)).collect();

        // First pass: computing node values
        // TODO handling F and QSmallType is inelegant; we might want to switch
        // to F for IO in NodeOps::prove
        let mut node_values = vec![output.clone()];
        let mut node_values_f = vec![output_f];

        for node in &self.nodes {
            output = node.padded_evaluate(output);
            let output_f: Vec<F> = output.values().iter().map(|x| F::from(*x)).collect();
            node_values.push(output.clone());
            node_values_f.push(output_f);
        }

        // Committing to node values
        let labeled_node_values: Vec<LabeledPolynomial<F, Poly<F>>> = node_values_f
            .iter()
            .map(|values|
            // TODO change dummy label once we e.g. have given numbers to the
            // nodes in the model: fc_1, fc_2, relu_1, etc.
            LabeledPolynomial::new(
                "dummy".to_string(),
                Poly::from_evaluations_vec(log2(values.len()) as usize, values.clone()),
                None,
                None,
            ))
            .collect();

        let (labeled_node_value_coms, node_value_coms_states) =
            PCS::commit(ck, &labeled_node_values, rng).unwrap();

        // Absorb all commitments into the sponge
        for lcom in labeled_node_value_coms.iter() {
            s.absorb(lcom.commitment());
        }

        // TODO Prove that all commited NIOs live in the right range (to be
        // discussed)

        let node_proofs = vec![];

        // Second pass: proving
        for ((((n, n_com), values), l_v_coms), v_coms_states) in self
            .nodes
            .iter()
            .zip(node_commitments.iter())
            .zip(node_values.windows(2))
            .zip(labeled_node_value_coms.windows(2))
            .zip(node_value_coms_states.windows(2))
        {
            // TODO prove likely needs to receive the sponge for randomness/FS
            let a = n.prove(
                s,
                n_com,
                values[0].clone(),
                l_v_coms[0].commitment(),
                values[1].clone(),
                l_v_coms[1].commitment(),
            );
        }

        // Opening model IO
        // TODO maybe this can be made more efficient by not committing to the
        // output nodes and instead working witht their plain values all along,
        // but that would require messy node-by-node handling
        let input_node = node_values.first().unwrap();
        let input_node_f = node_values_f.first().unwrap();
        let input_labeled_value = labeled_node_values.first().unwrap();
        let input_node_com = labeled_node_value_coms.first().unwrap();
        let input_node_com_state = node_value_coms_states.first().unwrap();

        let output_node = node_values.last().unwrap();
        let output_node_f = node_values_f.last().unwrap();
        let output_labeled_value = labeled_node_values.last().unwrap();
        let output_node_com = labeled_node_value_coms.last().unwrap();
        let output_node_com_state = node_value_coms_states.last().unwrap();

        // Absorb the model IO output and squeeze the challenge point
        // Absorb the plain output and squeeze the challenge point
        s.absorb(input_node_f);
        s.absorb(output_node_f);
        let input_challenge_point = s.squeeze_field_elements(log2(input_node_f.len()) as usize);
        let output_challenge_point = s.squeeze_field_elements(log2(output_node_f.len()) as usize);

        // TODO we have to pass rng, not None, but it has been moved before
        // fix this once we have decided how to handle the cumbersome
        // Option<&mut rng...>
        let input_opening_proof = PCS::open(
            ck,
            [input_labeled_value],
            [input_node_com],
            &input_challenge_point,
            s,
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
            s,
            [output_node_com_state],
            None,
        )
        .unwrap();

        /* TODO (important) Change output_node to all boundary nodes: first and last */
        // TODO prove that inputs match input commitments?
        InferenceProof {
            openings: vec![input_node.clone(), output_node.clone()],
            node_proofs,
            opening_proofs: vec![input_opening_proof, output_opening_proof],
        }
    }

    pub(crate) fn hide(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> (HiddenModel<F, S, PCS>, Vec<NodeCommitmentState<F, S, PCS>>) {
        // TODO blindly passing None, likely need to change to get hiding

        unimplemented!()
        // self.nodes.iter().map(|n| n.commit(ck, None)).collect()
    }
}
