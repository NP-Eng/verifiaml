use core::{borrow::Borrow, marker::PhantomData};

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
    sponge::{
        poseidon::{PoseidonConfig, PoseidonSponge},
        CryptographicSponge,
    },
};
use ark_ff::PrimeField;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{
    linear_codes::{LinearCodePCS, MultilinearBrakedown},
    to_bytes,
};
use ark_serialize::CanonicalSerialize;
use ark_std::{rand::RngCore, test_rng};
use blake2::{Blake2s256, Digest};
use sha2::Sha256;

pub(crate) fn test_sponge<F: PrimeField>() -> PoseidonSponge<F> {
    PoseidonSponge::new(&poseidon_parameters_for_test())
}

/// Generate default parameters for alpha = 17, state-size = 8
///
/// WARNING: This poseidon parameter is not secure. Please generate
/// your own parameters according the field you use.
fn poseidon_parameters_for_test<F: PrimeField>() -> PoseidonConfig<F> {
    let full_rounds = 8;
    let partial_rounds = 31;
    let alpha = 17;

    let mds = vec![
        vec![F::one(), F::zero(), F::one()],
        vec![F::one(), F::one(), F::zero()],
        vec![F::zero(), F::one(), F::one()],
    ];

    let mut ark = Vec::new();
    let mut ark_rng = test_rng();

    for _ in 0..(full_rounds + partial_rounds) {
        let mut res = Vec::new();

        for _ in 0..3 {
            res.push(F::rand(&mut ark_rng));
        }
        ark.push(res);
    }
    PoseidonConfig::new(full_rounds, partial_rounds, alpha, mds, ark, 2, 1)
}

#[cfg(test)]
pub(crate) struct LeafIdentityHasher;

#[cfg(test)]
impl CRHScheme for LeafIdentityHasher {
    type Input = Vec<u8>;
    type Output = Vec<u8>;
    type Parameters = ();

    fn setup<R: RngCore>(_: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        _: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        Ok(input.borrow().to_vec().into())
    }
}

#[cfg(test)]
pub(crate) struct FieldToBytesColHasher<F, D>
where
    F: PrimeField + CanonicalSerialize,
    D: Digest,
{
    _phantom: PhantomData<(F, D)>,
}

#[cfg(test)]
impl<F, D> CRHScheme for FieldToBytesColHasher<F, D>
where
    F: PrimeField + CanonicalSerialize,
    D: Digest,
{
    type Input = Vec<F>;
    type Output = Vec<u8>;
    type Parameters = ();

    fn setup<R: RngCore>(_rng: &mut R) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        _parameters: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut dig = D::new();
        dig.update(to_bytes!(input.borrow()).unwrap());
        Ok(dig.finalize().to_vec())
    }
}

pub(crate) type LeafH = LeafIdentityHasher;
pub(crate) type CompressH = Sha256;
pub(crate) type ColHasher<F, D> = FieldToBytesColHasher<F, D>;

pub(crate) struct MerkleTreeParams;

impl Config for MerkleTreeParams {
    type Leaf = Vec<u8>;

    type LeafDigest = <LeafH as CRHScheme>::Output;
    type LeafInnerDigestConverter = ByteDigestConverter<Self::LeafDigest>;
    type InnerDigest = <CompressH as TwoToOneCRHScheme>::Output;

    type LeafHash = LeafH;
    type TwoToOneHash = CompressH;
}

pub(crate) type MTConfig = MerkleTreeParams;
type Sponge<F> = PoseidonSponge<F>;

pub(crate) type BrakedownPCS<F> = LinearCodePCS<
    MultilinearBrakedown<
        F,
        MTConfig,
        Sponge<F>,
        DenseMultilinearExtension<F>,
        ColHasher<F, Blake2s256>,
    >,
    F,
    DenseMultilinearExtension<F>,
    Sponge<F>,
    MTConfig,
    ColHasher<F, Blake2s256>,
>;
