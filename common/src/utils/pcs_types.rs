use ark_crypto_primitives::{
    crh::{sha256::Sha256, CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{ByteDigestConverter, Config},
    sponge::poseidon::PoseidonSponge,
};
// no-std note:
// Currently, we use the `LeafIdentityHasher` from ark_pcs_bench_templates.
// This is not ideal, since the entire `ark_pcs_bench_templates` crate does not support `no_std`
// (due to `criterion`) dependency.
use ark_pcs_bench_templates::*;
use ark_poly::DenseMultilinearExtension;

use ark_poly_commit::linear_codes::{LinearCodePCS, MultilinearLigero};
use blake2::Blake2s256;

// Brakedown PCS over BN254
pub struct MerkleTreeParams;
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

type MTConfig = MerkleTreeParams;
type ColHasher<F> = FieldToBytesColHasher<F, Blake2s256>;

pub type Ligero<F> = LinearCodePCS<
    MultilinearLigero<F, MTConfig, PoseidonSponge<F>, DenseMultilinearExtension<F>, ColHasher<F>>,
    F,
    DenseMultilinearExtension<F>,
    PoseidonSponge<F>,
    MTConfig,
    ColHasher<F>,
>;
