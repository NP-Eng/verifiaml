use crate::NodeOpsVerify;
use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::Polynomial;
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

use hcs_common::{InferenceProof, InnerType, Model, NodeCommitment, Poly};

pub trait VerifyModel<F, S, PCS, ST, LT>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: Clone,
    LT: Clone,
{
    fn verify_inference(
        &self,
        vk: &PCS::VerifierKey,
        sponge: &mut S,
        node_commitments: &Vec<NodeCommitment<F, S, PCS>>,
        inference_proof: InferenceProof<F, S, PCS, ST, LT>,
    ) -> bool;
}

impl<F, S, PCS, ST, LT> VerifyModel<F, S, PCS, ST, LT> for Model<ST, LT>
where
    F: PrimeField + Absorb + From<ST> + From<LT>,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    ST: InnerType + TryFrom<LT>,
    LT: InnerType + From<ST>,
{
    fn verify_inference(
        &self,
        vk: &PCS::VerifierKey,
        sponge: &mut S,
        node_commitments: &Vec<NodeCommitment<F, S, PCS>>,
        inference_proof: InferenceProof<F, S, PCS, ST, LT>,
    ) -> bool {
        let InferenceProof {
            inputs,
            outputs,
            node_value_commitments,
            node_proofs,
            input_opening_proofs,
            output_opening_proofs,
        } = inference_proof;

        // Absorb all commitments into the sponge
        sponge.absorb(&node_value_commitments);

        // TODO Verify that all commited NIOs live in the right range (to be
        // discussed)

        // Verify node proofs
        for (((node, node_com), io_com), node_proof) in self
            .nodes
            .iter()
            .zip(node_commitments.iter())
            .zip(node_value_commitments.windows(2))
            .zip(node_proofs.into_iter())
        {
            if !node.verify(vk, sponge, node_com, &io_com[0], &io_com[1], node_proof) {
                return false;
            }
        }

        // Verifying model IO
        // TODO maybe this can be made more efficient by not committing to the
        // output nodes and instead working witht their plain values all along,
        // but that would require messy node-by-node handling
        let input_node_com = node_value_commitments.first().unwrap();
        let input_node_qarray = inputs[0].ref_small();
        let input_node_f: Vec<F> = input_node_qarray
            .values()
            .iter()
            .map(|x| F::from(*x))
            .collect();

        let output_node_com = node_value_commitments.last().unwrap();
        // TODO maybe it's better to save this as F in the proof?
        let output_node_f: Vec<F> = outputs[0]
            .ref_small()
            .values()
            .iter()
            .map(|x| F::from(*x))
            .collect();

        // Absorb the model IO output and squeeze the challenge point
        // Absorb the plain output and squeeze the challenge point
        sponge.absorb(&input_node_f);
        sponge.absorb(&output_node_f);
        let input_challenge_point =
            sponge.squeeze_field_elements(log2(input_node_f.len()) as usize);
        let output_challenge_point =
            sponge.squeeze_field_elements(log2(output_node_f.len()) as usize);

        // Verifying that the actual input was honestly padded with zeros
        let padded_input_shape = input_node_qarray.shape().clone();
        let honestly_padded_input = input_node_qarray
            .compact_resize(self.input_shape().clone(), ST::ZERO)
            .compact_resize(padded_input_shape, ST::ZERO);

        if honestly_padded_input.values() != input_node_qarray.values() {
            return false;
        }

        // The verifier must evaluate the MLE given by the plain input values
        let input_node_eval =
            Poly::from_evaluations_vec(log2(input_node_f.len()) as usize, input_node_f)
                .evaluate(&input_challenge_point);
        let output_node_eval =
            Poly::from_evaluations_vec(log2(output_node_f.len()) as usize, output_node_f)
                .evaluate(&output_challenge_point);

        // The computed values should match the openings of the corresponding
        // vectors
        // TODO rng, None
        if !PCS::check(
            vk,
            [input_node_com],
            &input_challenge_point,
            [input_node_eval],
            &input_opening_proofs[0],
            sponge,
            None,
        )
        .unwrap()
        {
            return false;
        }

        PCS::check(
            vk,
            [output_node_com],
            &output_challenge_point,
            [output_node_eval],
            &output_opening_proofs[0],
            sponge,
            None,
        )
        .unwrap()
    }
}
