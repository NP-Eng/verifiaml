use hcs_common::{
    quantise_f32_u8_nne, test_sponge, BMMNode, Ligero, Model, Node, Poly, QArray, QSmallType,
    QTypeArray, RequantiseBMMNode, ReshapeNode,
};
use hcs_prover::ProveModel;
use hcs_verifier::VerifyModel;

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

mod input;
mod parameters;

use ark_std::test_rng;
use input::*;
use parameters::*;

const INPUT_DIMS: &[usize] = &[28, 28];
const OUTPUT_DIMS: usize = 10;

// TODO this is incorrect now that we have switched to logs
fn build_simple_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let bmm: BMMNode<F, S, PCS> = BMMNode::new(
        WEIGHTS.to_vec(),
        BIAS.to_vec(),
        (flat_dim, OUTPUT_DIMS),
        Z_I,
    );

    let req_bmm: RequantiseBMMNode<F, S, PCS> =
        RequantiseBMMNode::new(OUTPUT_DIMS, S_I, Z_I, S_W, Z_W, S_O, Z_O);

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![
            Node::Reshape(reshape),
            Node::BMM(bmm),
            Node::RequantiseBMM(req_bmm),
        ],
    )
}

fn run_native_simple_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![135, 109, 152, 161, 187, 157, 159, 151, 173, 202];
    /**********************/

    let perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let output_i8 = perceptron.evaluate(input_i8);

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Output: {:?}", output_u8.values());
    assert_eq!(output_u8.move_values(), expected_output);
}

fn run_padded_simple_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![135, 109, 152, 161, 187, 157, 159, 151, 173, 202];
    /**********************/

    let perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let output_i8 = perceptron.padded_evaluate(input_i8);

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Output: {:?}", output_u8.values());
    assert_eq!(output_u8.move_values(), expected_output);
}

fn prove_inference_simple_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![135, 109, 152, 161, 187, 157, 159, 151, 173, 202];
    /**********************/

    let perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let mut rng = test_rng();
    let (ck, _) = perceptron.setup_keys(&mut rng).unwrap();

    let mut sponge: PoseidonSponge<Fr> = test_sponge();

    //let (hidden_nodes, com_states) = perceptron.commit(&ck, None).iter().unzip();
    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        perceptron.commit(&ck, None).into_iter().unzip();

    let inference_proof = perceptron.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.inputs_outputs[1].clone();

    let output_i8 = match output_qtypearray {
        QTypeArray::S(o) => o,
        _ => panic!("Expected QTypeArray::S"),
    };

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Padded output: {:?}", output_u8.values());
    assert_eq!(output_u8.move_values()[0..OUTPUT_DIMS], expected_output);
}

fn verify_inference_simple_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![135, 109, 152, 161, 187, 157, 159, 151, 173, 202];
    /**********************/

    let perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>();

    let quantised_input: QArray<u8> = input
        .iter()
        .map(|r| quantise_f32_u8_nne(r, S_INPUT, Z_INPUT))
        .collect::<Vec<Vec<u8>>>()
        .into();

    let input_i8 = (quantised_input.cast::<i32>() - 128).cast::<QSmallType>();

    let mut rng = test_rng();
    let (ck, vk) = perceptron.setup_keys(&mut rng).unwrap();

    let mut sponge: PoseidonSponge<Fr> = test_sponge();

    let (node_coms, node_com_states): (Vec<_>, Vec<_>) =
        perceptron.commit(&ck, None).into_iter().unzip();

    let inference_proof = perceptron.prove_inference(
        &ck,
        Some(&mut rng),
        &mut sponge,
        &node_coms,
        &node_com_states,
        input_i8,
    );

    let output_qtypearray = inference_proof.inputs_outputs[1].clone();

    let mut sponge: PoseidonSponge<Fr> = test_sponge();

    assert!(perceptron.verify_inference(&vk, &mut sponge, &node_coms, inference_proof));

    let output_i8 = match output_qtypearray {
        QTypeArray::S(o) => o,
        _ => panic!("Expected QTypeArray::S"),
    };

    let output_u8 = (output_i8.cast::<i32>() + 128).cast::<u8>();

    println!("Padded output: {:?}", output_u8.values());
    assert_eq!(output_u8.move_values()[0..OUTPUT_DIMS], expected_output);
}

fn main() {
    run_native_simple_perceptron_mnist();
    run_padded_simple_perceptron_mnist();
    prove_inference_simple_perceptron_mnist();
    verify_inference_simple_perceptron_mnist();
}
