use ark_std::{log2, rand::RngCore};

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{LabeledPolynomial, PolynomialCommitment};

use crate::{
    error::VerificationError, hidden_model::hidden_nodes::HiddenNodeOps, proofs::InferenceProof,
    Poly,
};

use self::hidden_nodes::HiddenNode;

pub(crate) mod hidden_nodes;

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all nodes to use the same PCS; this might change
// in the future
pub(crate) struct HiddenModel<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    input_shape_log: Vec<usize>,
    output_shape_log: Vec<usize>,
    nodes: Vec<HiddenNode<F, S, PCS>>,
}

impl<F, S, PCS> HiddenModel<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    pub(crate) fn new(input_shape_log: Vec<usize>, nodes: Vec<HiddenNode<F, S, PCS>>) -> Self {
        // An empty model would cause problems later on
        assert!(!nodes.is_empty(), "A model cannot have no nodes",);

        Self {
            input_shape_log,
            output_shape_log: nodes.last().unwrap().padded_shape_log(),
            nodes,
        }
    }

    pub(crate) fn get_nodes(&self) -> &Vec<HiddenNode<F, S, PCS>> {
        &self.nodes
    }

    // pub(crate) fn verify_inference(
    //     &self,
    //     vk: &PCS::VerifierKey,
    //     proof: InferenceProof<F, S, PCS>,
    // ) -> Result<(), VerificationError> {
    //     // TODO Absorb public parameters into s (to be determined what exactly)

    //     let mut output = input.compact_resize(
    //         self.input_shape
    //             .iter()
    //             .map(|x| x.next_power_of_two())
    //             .collect(),
    //         0,
    //     );

    //     let output_f = output.values().iter().map(|x| F::from(*x)).collect();

    //     // First pass: computing node values
    //     // TODO handling F and QSmallType is inelegant; we might want to switch
    //     // to F for IO in NodeOps::prove
    //     let mut node_values = vec![output.clone()];
    //     let mut node_values_f = vec![output_f];

    //     for node in &self.nodes {
    //         output = node.padded_evaluate(output);
    //         let output_f: Vec<F> = output.values().iter().map(|x| F::from(*x)).collect();
    //         node_values.push(output.clone());
    //         node_values_f.push(output_f);
    //     }

    //     // Committing to node values
    //     let labeled_node_values: Vec<LabeledPolynomial<F, Poly<F>>> = node_values_f
    //         .iter()
    //         .map(|values|
    //         // TODO change dummy label once we e.g. have given numbers to the
    //         // nodes in the model: fc_1, fc_2, relu_1, etc.
    //         LabeledPolynomial::new(
    //             "dummy".to_string(),
    //             Poly::from_evaluations_vec(log2(values.len()) as usize, values.clone()),
    //             None,
    //             None,
    //         ))
    //         .collect();

    //     let (labeled_node_value_coms, node_value_coms_states) =
    //         PCS::commit(ck, &labeled_node_values, rng).unwrap();

    //     // Absorb all commitments into the sponge
    //     for lcom in labeled_node_value_coms.iter() {
    //         s.absorb(lcom.commitment());
    //     }

    //     // TODO Prove that all commited NIOs live in the right range (to be
    //     // discussed)

    //     let node_proofs = vec![];

    //     // Second pass: proving
    //     for ((((n, n_com), values), l_v_coms), v_coms_states) in self
    //         .nodes
    //         .iter()
    //         .zip(node_commitments.iter())
    //         .zip(node_values.windows(2))
    //         .zip(labeled_node_value_coms.windows(2))
    //         .zip(node_value_coms_states.windows(2))
    //     {
    //         // TODO prove likely needs to receive the sponge for randomness/FS
    //         let a = n.prove(
    //             s,
    //             n_com,
    //             values[0].clone(),
    //             l_v_coms[0].commitment(),
    //             values[1].clone(),
    //             l_v_coms[1].commitment(),
    //         );
    //     }

    //     // Opening model IO
    //     // TODO maybe this can be made more efficient by not committing to the
    //     // output nodes and instead working witht their plain values all along,
    //     // but that would require messy node-by-node handling
    //     let input_node = node_values.first().unwrap();
    //     let input_node_f = node_values_f.first().unwrap();
    //     let input_labeled_value = labeled_node_values.first().unwrap();
    //     let input_node_com = labeled_node_value_coms.first().unwrap();
    //     let input_node_com_state = node_value_coms_states.first().unwrap();

    //     let output_node = node_values.last().unwrap();
    //     let output_node_f = node_values_f.last().unwrap();
    //     let output_labeled_value = labeled_node_values.last().unwrap();
    //     let output_node_com = labeled_node_value_coms.last().unwrap();
    //     let output_node_com_state = node_value_coms_states.last().unwrap();

    //     // Absorb the model IO output and squeeze the challenge point
    //     // Absorb the plain output and squeeze the challenge point
    //     s.absorb(input_node_f);
    //     s.absorb(output_node_f);
    //     let input_challenge_point = s.squeeze_field_elements(log2(input_node_f.len()) as usize);
    //     let output_challenge_point = s.squeeze_field_elements(log2(output_node_f.len()) as usize);

    //     // TODO we have to pass rng, not None, but it has been moved before
    //     // fix this once we have decided how to handle the cumbersome
    //     // Option<&mut rng...>
    //     let input_opening_proof = PCS::open(
    //         ck,
    //         [input_labeled_value],
    //         [input_node_com],
    //         &input_challenge_point,
    //         s,
    //         [input_node_com_state],
    //         None,
    //     )
    //     .unwrap();

    //     // TODO we have to pass rng, not None, but it has been moved before
    //     // fix this once we have decided how to handle the cumbersome
    //     // Option<&mut rng...>
    //     let output_opening_proof = PCS::open(
    //         ck,
    //         [output_labeled_value],
    //         [output_node_com],
    //         &output_challenge_point,
    //         s,
    //         [output_node_com_state],
    //         None,
    //     )
    //     .unwrap();

    //     /* TODO (important) Change output_node to all boundary nodes: first and last */
    //     // TODO prove that inputs match input commitments?
    //     InferenceProof {
    //         outputs: vec![input_node.clone(), output_node.clone()],
    //         node_proofs,
    //         opening_proofs: vec![input_opening_proof, output_opening_proof],
    //     }
    // }
}
