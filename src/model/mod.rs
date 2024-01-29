use std::marker::PhantomData;

use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, Polynomial};
use ark_poly_commit::PolynomialCommitment;
use ark_std::log2;

use crate::model::nodes::Node;

mod nodes;

pub(crate) type Poly<F> = DenseMultilinearExtension<F>;

// TODO change the functions that receive vectors to receive slices instead whenever it makes sense

// TODO: for now, we require all layers to use the same PCS; this might change
// in the future
pub struct Model<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    layers: Vec<Node<F, S, PCS>>,
    phantom: PhantomData<(F, S, PCS)>,
}

impl<F, S, PCS> Model<F, S, PCS>
where
    F: PrimeField,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
{
    fn new(layers: Vec<Node<F, S, PCS>>) -> Self {
        Self {
            layers,
            phantom: PhantomData,
        }
    }
}
