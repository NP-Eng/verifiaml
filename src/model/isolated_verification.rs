use std::vec;

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

fn verify_model<F, S, PCS>(
    vk: &PCS::VerifierKey,
    model: &Model<F, S, PCS>,
    node_commitments: &Vec<NodeCommitment<F, S, PCS>>,
    inference_proof: InferenceProof<F, S, PCS>,
    sponge: &mut S,
) where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let input_node = model.nodes.first().unwrap();
    let output_node = model.nodes.last().unwrap();

    // Absorb all commitments into the sponge
    sponge.absorb(inference_proof);

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
            &node_com,
            &node_com_state,
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
    let input_node = node_outputs.first().unwrap();
    let input_node_f = node_output_mles.first().unwrap().to_evaluations();
    let input_labeled_value = labeled_output_mles.first().unwrap();
    let input_node_com = output_coms.first().unwrap();
    let input_node_com_state = output_com_states.first().unwrap();

    let output_node = node_outputs.last().unwrap();
    let output_node_f = node_output_mles.last().unwrap().to_evaluations();
    let output_labeled_value = labeled_output_mles.last().unwrap();
    let output_node_com = output_coms.last().unwrap();
    let output_node_com_state = output_com_states.last().unwrap();

    // Absorb the model IO output and squeeze the challenge point
    // Absorb the plain output and squeeze the challenge point
    sponge.absorb(&input_node_f);
    sponge.absorb(&output_node_f);
    let input_challenge_point = sponge.squeeze_field_elements(log2(input_node_f.len()) as usize);
    let output_challenge_point = sponge.squeeze_field_elements(log2(output_node_f.len()) as usize);

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
        inputs_outputs: vec![input_node.clone(), output_node.clone()],
        node_proofs,
        opening_proofs: vec![input_opening_proof, output_opening_proof],
    }
}
