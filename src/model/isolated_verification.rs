use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_sumcheck::ml_sumcheck::{
    protocol::{verifier::SubClaim, PolynomialInfo},
    MLSumcheck,
};

use crate::model::nodes::bmm::{BMMNodeCommitment, BMMNodeProof};

use super::{
    nodes::{NodeCommitment, NodeProof},
    Poly,
};

fn verify_bmm_node<F, S, PCS>(
    vk: &PCS::VerifierKey,
    sponge: &mut S,
    node_com: &NodeCommitment<F, S, PCS>,
    input_com: &LabeledCommitment<PCS::Commitment>,
    output_com: &LabeledCommitment<PCS::Commitment>,
    proof: NodeProof<F, S, PCS>,
    padded_dims_log: (usize, usize),
) -> bool
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
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
        bias_opening_proof,
        bias_opening_value,
        output_opening_proof,
        output_opening_value,
    } = match proof {
        NodeProof::BMM(p) => p,
        _ => panic!("Expected BMMNodeProof"),
    };

    let r: Vec<F> = sponge.squeeze_field_elements(padded_dims_log.1);

    // The value proved in sumcheck should be the difference between the output
    // and the bias
    let sumcheck_evaluation = output_opening_value - bias_opening_value;

    // Information about the polynomial f(s) = input_mle(s) * weight_mle(s, r)
    // to which sumcheck is applied
    let info = PolynomialInfo {
        max_multiplicands: 2,
        num_variables: padded_dims_log.0,
        products: vec![(F::one(), vec![0, 1])],
    };

    let Ok(subclaim) = MLSumcheck::verify(&info, sumcheck_evaluation, &sumcheck_proof, sponge)
    else {
        return false;
    };

    let SubClaim {
        point: oracle_point,
        expected_evaluation: oracle_evaluation,
    } = subclaim;

    if oracle_evaluation != input_opening_value * weight_opening_value {
        return false;
    }

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

    // TODO possibly rng, not None
    if !PCS::check(
        vk,
        [weight_com],
        &oracle_point
            .into_iter()
            .chain(r.clone().into_iter())
            .collect(),
        [weight_opening_value],
        &weight_opening_proof,
        sponge,
        None,
    )
    .unwrap()
    {
        return false;
    }

    if !PCS::check(
        vk,
        [bias_com],
        &r,
        [bias_opening_value],
        &bias_opening_proof,
        sponge,
        None,
    )
    .unwrap()
    {
        return false;
    }

    PCS::check(
        vk,
        [output_com],
        &r,
        [output_opening_value],
        &output_opening_proof,
        sponge,
        None,
    )
    .unwrap()
}
