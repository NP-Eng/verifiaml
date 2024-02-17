use crate::{
    model::{
        nodes::{loose_fc::LooseFCNode, reshape::ReshapeNode, Node, NodeType},
        qarray::QArray,
        Model, Poly,
    },
    pcs_types::Brakedown,
    quantization::{quantise_f32_u8_nne, QSmallType},
};

use ark_bn254::Fr;
use ark_crypto_primitives::sponge::{poseidon::PoseidonSponge, Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

mod input;
mod parameters;

use input::*;
use parameters::*;

const INPUT_DIMS: &[usize] = &[28, 28];
const OUTPUT_DIMS: &[usize] = &[10];

// TODO this is incorrect now that we have switched to logs
fn build_simple_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS>
where
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> =
        ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim], NodeType::Input);

    let lfc: LooseFCNode<F, S, PCS> = LooseFCNode::new(
        WEIGHTS.to_vec(),
        BIAS.to_vec(),
        (flat_dim, OUTPUT_DIMS[0]),
        (INPUT_DIMS[0], INPUT_DIMS[1]),
        S_I,
        Z_I,
        S_W,
        Z_W,
        S_O,
        Z_O,
        NodeType::Output,
    );

    Model::new(
        INPUT_DIMS.to_vec(),
        vec![Node::Reshape(reshape), Node::LooseFC(lfc)],
    )
}

#[test]
fn run_simple_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![135, 109, 152, 161, 187, 157, 159, 151, 173, 202];
    /**********************/

    let perceptron = build_simple_perceptron_mnist::<Fr, PoseidonSponge<Fr>, Brakedown<Fr>>();

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
