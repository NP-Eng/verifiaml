
use ark_ff::PrimeField;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_poly_commit::PolynomialCommitment;
use crate::model::Poly;

use crate::model::Model;

mod parameters;

const INPUT_DIMS: &[usize] = &[28, 28];

const OUTPUT_DIMS: &[usize] = &[10];

fn build_simple_perceptron_mnist<F, S, PCS>() -> Model<F, S, PCS> 
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{


    let mut nodes = Vec::new();

}