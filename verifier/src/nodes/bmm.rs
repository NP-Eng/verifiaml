use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_sumcheck::ml_sumcheck::{
    protocol::{verifier::SubClaim, PolynomialInfo},
    MLSumcheck,
};
use hcs_common::{BMMNode, BMMNodeCommitment, BMMNodeProof, NodeCommitment, NodeProof, Poly};

use crate::NodeOpsVerify;

impl<F, S, PCS> NodeOpsVerify<F, S, PCS> for BMMNode<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn verify(
        &self,
        vk: &PCS::VerifierKey,
        sponge: &mut S,
        node_com: &NodeCommitment<F, S, PCS>,
        input_com: &LabeledCommitment<PCS::Commitment>,
        output_com: &LabeledCommitment<PCS::Commitment>,
        proof: NodeProof<F, S, PCS>,
    ) -> bool {
        let NodeCommitment::BMM(BMMNodeCommitment {
            weight_com,
            bias_com,
        }) = node_com
        else {
            panic!("Expected BMMNodeCommitment")
        };

        let BMMNodeProof {
            sumcheck_proof,
            input_opening_proof,
            input_opening_value,
            weight_opening_proof,
            weight_opening_value,
            output_bias_opening_proof,
            output_opening_value,
            bias_opening_value,
        } = match proof {
            NodeProof::BMM(p) => p,
            _ => panic!("Expected BMMNodeProof"),
        };

        // Squeezing random challenge r to bind the first variables of W^ to
        let r: Vec<F> = sponge.squeeze_field_elements(self.padded_dims_log.1);

        // The hypercube sum proved in sumcheck should be the difference between
        // the output and the bias
        let sumcheck_evaluation = output_opening_value - bias_opening_value;

        // Public information about the sumchecked polynomial
        // g(x) = (input - zero_point)^(x) * W^(r, x),
        let info = PolynomialInfo {
            max_multiplicands: 2,
            num_variables: self.padded_dims_log.0,
            products: vec![(F::one(), vec![0, 1])],
        };

        // Verify the sumcheck proof for g and obtaining the oracle-call point s
        // and claimed evaluation g(s)
        let Ok(subclaim) = MLSumcheck::verify(&info, sumcheck_evaluation, &sumcheck_proof, sponge)
        else {
            return false;
        };

        let SubClaim {
            point: oracle_point,
            expected_evaluation: oracle_evaluation,
        } = subclaim;

        // Verify g(s) agrees with the claims for (input - zero_point)^(s) and
        // W^(r, s)
        if oracle_evaluation
            != (input_opening_value - F::from(self.input_zero_point)) * weight_opening_value
        {
            return false;
        }

        // Verify that the opening of input^ at s agrees with the claimed value for
        // (input - zero_point)^(s)
        // TODO possibly rng, not None
        if !PCS::check(
            vk,
            [input_com],
            &oracle_point,
            [input_opening_value],
            &input_opening_proof,
            sponge,
            None,
        )
        .unwrap()
        {
            return false;
        }

        // Verify the openings of W^ at r || s and b and o at r match the claimed
        // values
        // TODO possibly rng, not None
        if !PCS::check(
            vk,
            [weight_com],
            &r.clone().into_iter().chain(oracle_point).collect(),
            [weight_opening_value],
            &weight_opening_proof,
            sponge,
            None,
        )
        .unwrap()
        {
            return false;
        }

        PCS::check(
            vk,
            [output_com, bias_com],
            &r,
            [output_opening_value, bias_opening_value],
            &output_bias_opening_proof,
            sponge,
            None,
        )
        .unwrap()
    }
}
