use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly_commit::PolynomialCommitment;

use crate::{
    model::{
        nodes::{fc::FCNode, loose_fc::LooseFCNode, relu::ReLUNode, reshape::ReshapeNode, Node},
        qarray::QArray,
        Model, Poly,
    },
    quantization::{quantise_f32_u8_nne, QSmallType},
};

mod input;
mod parameters;

use input::*;
use parameters::*;

use ark_bn254::{Fr, G1Affine};
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_poly_commit::hyrax::HyraxPC;

type Sponge = PoseidonSponge<Fr>;
type Hyrax254 = HyraxPC<G1Affine, Poly<Fr>, Sponge>;

const INPUT_DIMS: &[usize] = &[28, 28];
const INTER_DIM: usize = 28;
const OUTPUT_DIM: usize = 10;

// TODO this is incorrect now that we have switched to logs
fn build_two_layer_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{

    let flat_dim = INPUT_DIMS.iter().product();

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

    let lfc: LooseFCNode<F, S, PCS> = LooseFCNode::new(
        WEIGHTS_1.to_vec(),
        BIAS_1.to_vec(),
        (flat_dim, INTER_DIM),
        (INPUT_DIMS[0], INPUT_DIMS[1]),
        S_1_I,
        Z_1_I,
        S_1_W,
        Z_1_W,
        S_1_O,
        Z_1_O,
    );

    // let fc1: FCNode<F, S, PCS> = FCNode::new(
    //     WEIGHTS_1.to_vec(),
    //     BIAS_1.to_vec(),
    //     (flat_dim, INTER_DIM),
    //     S_1_I,
    //     Z_1_I,
    //     S_1_W,
    //     Z_1_W,
    //     S_1_O,
    //     Z_1_O,
    // );

    let relu: ReLUNode<F, S, PCS> = ReLUNode::new(28, Z_1_O);

    let fc2: FCNode<F, S, PCS> = FCNode::new(
        WEIGHTS_2.to_vec(),
        BIAS_2.to_vec(),
        (INTER_DIM, OUTPUT_DIM),
        S_2_I,
        Z_2_I,
        S_2_W,
        Z_2_W,
        S_2_O,
        Z_2_O,
    );

    Model::new(INPUT_DIMS.to_vec(), vec![
        Node::Reshape(reshape),
        Node::LooseFC(lfc),
        Node::ReLU(relu),
        Node::FC(fc2),
    ])
}

#[test]
fn run_two_layer_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![138, 106, 149, 160, 174, 152, 141, 146, 169, 207];
    /**********************/

    let perceptron = build_two_layer_perceptron_mnist::<Fr, Sponge, Hyrax254>();

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
