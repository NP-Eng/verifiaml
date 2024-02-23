use std::vec;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, Polynomial};
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use ark_std::log2;
use ark_sumcheck::ml_sumcheck::{
    protocol::{verifier::SubClaim, PolynomialInfo},
    MLSumcheck,
};

use crate::model::nodes::bmm::{BMMNodeCommitment, BMMNodeProof};

use super::{
    nodes::{Node, NodeCommitment, NodeProof},
    qarray::{QArray, QTypeArray},
    InferenceProof, Model, Poly,
};

fn verify_bmm_node<F, S, PCS>(
    vk: &PCS::VerifierKey,
    sponge: &mut S,
    node_com: &NodeCommitment<F, S, PCS>,
    input_com: &LabeledCommitment<PCS::Commitment>,
    output_com: &LabeledCommitment<PCS::Commitment>,
    proof: NodeProof<F, S, PCS>,
    padded_dims_log: (usize, usize),
    input_zero_point: F, // This argument will not be here in the final code
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

    if oracle_evaluation != (input_opening_value - input_zero_point) * weight_opening_value {
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

fn verify_node<F, S, PCS>(
    vk: &PCS::VerifierKey,
    sponge: &mut S,
    node_com: &NodeCommitment<F, S, PCS>,
    input_com: &LabeledCommitment<PCS::Commitment>,
    output_com: &LabeledCommitment<PCS::Commitment>,
    proof: NodeProof<F, S, PCS>,
    padded_dims_log: Option<(usize, usize)>,
    input_zero_point: Option<F>,
) -> bool
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    match node_com {
        NodeCommitment::BMM(_) => verify_bmm_node(
            vk,
            sponge,
            node_com,
            input_com,
            output_com,
            proof,
            padded_dims_log.unwrap(),
            input_zero_point.unwrap(),
        ),
        _ => true,
    }
}

pub(crate) fn verify_inference<F, S, PCS>(
    vk: &PCS::VerifierKey,
    sponge: &mut S,
    model: &Model<F, S, PCS>,
    node_commitments: &Vec<NodeCommitment<F, S, PCS>>,
    inference_proof: InferenceProof<F, S, PCS>,
) -> bool
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let InferenceProof {
        inputs_outputs,
        node_value_commitments,
        node_proofs,
        opening_proofs,
    } = inference_proof;

    // Absorb all commitments into the sponge
    sponge.absorb(&node_value_commitments);

    // TODO Verify that all commited NIOs live in the right range (to be
    // discussed)

    // Verify node proofs
    for (((node, node_com), io_com), node_proof) in model
        .nodes
        .iter()
        .zip(node_commitments.iter())
        .zip(node_value_commitments.windows(2))
        .zip(node_proofs.into_iter())
    {
        // This will not be necessary in the actual code, as the BMM dimensions
        // and zero point  will be contained in the (possibly hidden) BMMNode
        // and therefore won't be passed to the proving method
        let (padded_dims_log, input_zero_point) = match node {
            Node::BMM(bmm) => (
                Some(bmm.padded_dims_log()),
                Some(F::from(bmm.input_zero_point())),
            ),
            _ => (None, None),
        };

        if !verify_node(
            vk,
            sponge,
            node_com,
            &io_com[0],
            &io_com[1],
            node_proof,
            padded_dims_log,
            input_zero_point,
        ) {
            return false;
        }
    }

    // Verifying model IO
    // TODO maybe this can be made more efficient by not committing to the
    // output nodes and instead working witht their plain values all along,
    // but that would require messy node-by-node handling
    let input_node_com = node_value_commitments.first().unwrap();
    let input_node_qarray = match &inputs_outputs[0] {
        QTypeArray::S(i) => i,
        _ => panic!("Model input should be QTypeArray::S"),
    };
    let input_node_f: Vec<F> = input_node_qarray
        .values()
        .iter()
        .map(|x| F::from(*x))
        .collect();

    let output_node_com = node_value_commitments.last().unwrap();
    // TODO maybe it's better to save this as F in the proof?
    let output_node_f: Vec<F> = match &inputs_outputs[1] {
        QTypeArray::S(o) => o.values().iter().map(|x| F::from(*x)).collect(),
        _ => panic!("Model output should be QTypeArray::S"),
    };

    // Absorb the model IO output and squeeze the challenge point
    // Absorb the plain output and squeeze the challenge point
    sponge.absorb(&input_node_f);
    sponge.absorb(&output_node_f);
    let input_challenge_point = sponge.squeeze_field_elements(log2(input_node_f.len()) as usize);
    let output_challenge_point = sponge.squeeze_field_elements(log2(output_node_f.len()) as usize);

    // Verifying that the actual input was honestly padded with zeros
    let padded_input_shape = input_node_qarray.shape().clone();
    let honestly_padded_input = input_node_qarray
        .compact_resize(model.input_shape().clone(), 0)
        .compact_resize(padded_input_shape, 0);

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

    // TODO rng, None
    if !PCS::check(
        vk,
        [input_node_com],
        &input_challenge_point,
        [input_node_eval],
        &opening_proofs[0],
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
        &opening_proofs[1],
        sponge,
        None,
    )
    .unwrap()
}
