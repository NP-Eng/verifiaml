use hcs_common::NodeOpsNative;

use hcs_common::{
    BMMNode, LabeledPoly, NodeCommitment, NodeCommitmentState, NodeOpsPadded, NodeProof, Poly, ReLUNode, SmallNIO, ReshapeNode, RequantizeBMMFloatNode

};
use ark_bn254::Fr;
use hcs_common::Ligero;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_std::any::Any;

use ark_crypto_primitives::sponge::{Absorb, CryptographicSponge};
use ark_ff::PrimeField;
use ark_poly_commit::{LabeledCommitment, PolynomialCommitment};
use crate::NodeOpsProve;

pub trait DowncastNode<F, S, PCS, N, ST> 
where 
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
    N: NodeOpsNative<ST> + 'static,
    ST: SmallNIO,
{
    fn downcast_as<T: NodeOpsProve<F, S, PCS, i8> + 'static>(&self) -> Option<&dyn NodeOpsProve<F, S, PCS, ST>>;
    fn downcast(&self) -> &dyn NodeOpsProve<F, S, PCS, ST>;
}

macro_rules! downcast_node_impl {
    ($st:ty) => {
        impl<F, S, PCS, N> DowncastNode<F, S, PCS, N, $st> for Box<N>
        where
            F: PrimeField + Absorb,
            S: CryptographicSponge,
            PCS: PolynomialCommitment<F, Poly<F>, S>,
            N: NodeOpsNative<$st> + 'static
        { 
            fn downcast_as<T: NodeOpsProve<F, S, PCS, $st> + 'static>(&self) -> Option<&dyn NodeOpsProve<F, S, PCS, $st>> {
                (self as &dyn Any).downcast_ref::<T>().map(|x| x as &dyn NodeOpsProve<F, S, PCS, $st>)
            }

            // Downcast to any node type in the model
            fn downcast(&self) -> &dyn NodeOpsProve<F, S, PCS, $st> { 
                self.downcast_as::<ReLUNode<$st>>() 
                    .or_else(|| self.downcast_as::<BMMNode<$st>>())
                    .or_else(|| self.downcast_as::<RequantizeBMMFloatNode<$st>>())
                    .or_else(|| self.downcast_as::<ReshapeNode>())
                    .unwrap()
            }
        }
    };
}

downcast_node_impl!(i8);


#[test]
fn test_downcast_relu() {
    let node: Box<dyn NodeOpsNative<i8>> = Box::new(ReLUNode::new(1, 1));
    let prove_node: &dyn NodeOpsProve<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8> = node.downcast();
    assert_eq!(prove_node.type_name(), "ReLU");
}