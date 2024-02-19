use ark_std::{log2, rand::RngCore};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};

use crate::model::nodes::{NodeOps, NodeOpsSNARK};
use crate::{model::nodes::Node, quantization::QSmallType};

use self::qarray::QTypeArray;
use self::{
    nodes::{NodeCommitment, NodeCommitmentState, NodeProof},
    qarray::QArray,
};

mod examples;
mod nodes;
mod qarray;
mod reshaping;

pub(crate) type Poly<F> = DenseMultilinearExtension<F>;

pub(crate) struct InferenceProof<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    // Model output tensors
    outputs: Vec<QTypeArray>,

    // Proofs of evaluation of each of the model's nodes
    node_proofs: Vec<NodeProof>,

    // Proofs of opening of each of the model's outputs
    opening_proofs: Vec<PCS::Proof>,
}

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
        let mut output = QTypeArray::S(input);
        for node in &self.nodes {
            output = node.evaluate(output);
        }

        if let QTypeArray::S(output) = output {
            output
        } else {
            panic!("Output QArray type should be QSmallType")
        }
    }

    /// Unlike the node's `padded_evaluate`, the model's `padded_evaluate` accepts unpadded input
    /// and first re-sizes it before running inference.
    pub(crate) fn padded_evaluate(&self, input: QArray<QSmallType>) -> QArray<QSmallType> {
        // TODO sanity check: input shape matches model input shape

        let input = input.compact_resize(
            // TODO this functionality is so common we might as well make it an #[inline] function
            self.input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            0,
        );

        let mut output = QTypeArray::S(input);

        for node in &self.nodes {
            output = node.padded_evaluate(&output);
        }

        // TODO switch to reference in reshape?

        if let QTypeArray::S(output) = output {
            output.compact_resize(self.output_shape.clone(), 0)
        } else {
            panic!("Output QArray type should be QSmallType")
        }
    }

    pub(crate) fn prove_inference(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
        sponge: &mut S,
        node_commitments: Vec<NodeCommitment<F, S, PCS>>,
        input: QArray<QSmallType>,
    ) -> InferenceProof<F, S, PCS> {
        // TODO Absorb public parameters into s (to be determined what exactly)

        let output = input.compact_resize(
            self.input_shape
                .iter()
                .map(|x| x.next_power_of_two())
                .collect(),
            0,
        );

        let output_f = output.values().iter().map(|x| F::from(*x)).collect();

        let mut output = QTypeArray::S(output);

        // First pass: computing node values
        // TODO handling F and QSmallType is inelegant; we might want to switch
        // to F for IO in NodeOps::prove
        let mut node_outputs = vec![output.clone()];
        let mut node_outputs_f = vec![output_f];

        for node in &self.nodes {
            output = node.padded_evaluate(&output);

            let output_f: Vec<F> = match &output {
                QTypeArray::S(o) => o.values().iter().map(|x| F::from(*x)).collect(),
                QTypeArray::L(o) => o.values().iter().map(|x| F::from(*x)).collect(),
            };

            node_outputs.push(output.clone());
            node_outputs_f.push(output_f);
        }

        // Committing to node outputs as MLEs (individual per node for now)
        let output_mles: Vec<LabeledPolynomial<F, Poly<F>>> = node_outputs_f
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

        let (node_coms, node_com_states) = PCS::commit(ck, &output_mles, rng).unwrap();

        // Absorb all commitments into the sponge
        sponge.absorb(&node_coms);

        // TODO Prove that all commited NIOs live in the right range (to be
        // discussed)

        let mut node_proofs = Vec::new();

        // Second pass: proving
        for ((((node, node_com), values), l_v_coms), v_coms_states) in self
            .nodes
            .iter()
            .zip(node_commitments.iter())
            .zip(node_outputs.windows(2))
            .zip(node_coms.windows(2))
            .zip(node_com_states.windows(2))
        {
            // TODO prove likely needs to receive the sponge for randomness/FS
            node_proofs.push(node.prove(
                sponge,
                node_com,
                values[0].clone(),
                l_v_coms[0].commitment(),
                values[1].clone(),
                l_v_coms[1].commitment(),
            ));
        }

        // Opening model IO
        // TODO maybe this can be made more efficient by not committing to the
        // output nodes and instead working witht their plain values all along,
        // but that would require messy node-by-node handling
        let input_node = node_outputs.first().unwrap();
        let input_node_f = node_outputs_f.first().unwrap();
        let input_labeled_value = output_mles.first().unwrap();
        let input_node_com = node_coms.first().unwrap();
        let input_node_com_state = node_com_states.first().unwrap();

        let output_node = node_outputs.last().unwrap();
        let output_node_f = node_outputs_f.last().unwrap();
        let output_labeled_value = output_mles.last().unwrap();
        let output_node_com = node_coms.last().unwrap();
        let output_node_com_state = node_com_states.last().unwrap();

        // Absorb the model IO output and squeeze the challenge point
        // Absorb the plain output and squeeze the challenge point
        sponge.absorb(input_node_f);
        sponge.absorb(output_node_f);
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

        /* TODO (important) Change output_node to all boundary nodes: first and last */
        // TODO prove that inputs match input commitments?
        InferenceProof {
            outputs: vec![input_node.clone(), output_node.clone()],
            node_proofs,
            opening_proofs: vec![input_opening_proof, output_opening_proof],
        }
    }

    pub(crate) fn commit(
        &self,
        ck: &PCS::CommitterKey,
        rng: Option<&mut dyn RngCore>,
    ) -> Vec<(NodeCommitment<F, S, PCS>, NodeCommitmentState<F, S, PCS>)> {
        // TODO blindly passing None, likely need to change to get hiding
        self.nodes.iter().map(|n| n.commit(ck, None)).collect()
    }
}
