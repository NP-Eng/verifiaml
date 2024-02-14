
use crate::{
    model::{
        nodes::{loose_fc::LooseFCNode, reshape::ReshapeNode, Node},
        qarray::QArray,
        Model, Poly,
    },
    quantization::{quantise_f32_u8_nne, QSmallType},
};

use ark_crypto_primitives::{
    crh::{sha256::Sha256, CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
    sponge::{poseidon::PoseidonSponge, Absorb, CryptographicSponge},
};
use ark_pcs_bench_templates::*;
use ark_poly::DenseMultilinearExtension;

use ark_bn254::Fr;
use ark_ff::PrimeField;

use ark_poly_commit::{linear_codes::{LinearCodePCS, MultilinearBrakedown}, PolynomialCommitment};
use blake2::Blake2s256;

mod input;
mod parameters;

// Brakedown PCS over BN254
struct MerkleTreeParams;
type LeafH = LeafIdentityHasher;
type CompressH = Sha256;
impl Config for MerkleTreeParams {
    type Leaf = Vec<u8>;

    type LeafDigest = <LeafH as CRHScheme>::Output;
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

    type LeafHash = LeafH;
    type TwoToOneHash = CompressH;
}

pub type MLE<F> = DenseMultilinearExtension<F>;
type MTConfig = MerkleTreeParams;
type Sponge<F> = PoseidonSponge<F>;
type ColHasher<F> = FieldToBytesColHasher<F, Blake2s256>;
type Brakedown<F> = LinearCodePCS<
    MultilinearBrakedown<F, MTConfig, Sponge<F>, MLE<F>, ColHasher<F>>,
    F,
    MLE<F>,
    Sponge<F>,
    MTConfig,
    ColHasher<F>,
>;

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

    let reshape: ReshapeNode<F, S, PCS> = ReshapeNode::new(INPUT_DIMS.to_vec(), vec![flat_dim]);

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
    );

    Model::new(INPUT_DIMS.to_vec(), vec![Node::Reshape(reshape), Node::LooseFC(lfc)])
}

#[test]
fn run_simple_perceptron_mnist() {
    /**** Change here ****/
    let input = NORMALISED_INPUT_TEST_150;
    let expected_output: Vec<u8> = vec![135, 109, 152, 161, 187, 157, 159, 151, 173, 202];
    /**********************/

    let perceptron = build_simple_perceptron_mnist::<Fr, Sponge<Fr>, Brakedown<Fr>>();

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
