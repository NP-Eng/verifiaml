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
use rayon::vec;
use crate::NodeOpsProve;

/* pub trait DowncastNode<F, S, PCS, N, ST> 
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
        impl<F, S, PCS, N> DowncastNode<F, S, PCS, N, $st> for N
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

downcast_node_impl!(i8); */

// error[E0605]: non-primitive cast: `&dyn NodeOpsNative<i8>` as `&(dyn Any + 'static)`
fn node_downcast<
    F: PrimeField + Absorb,
    S: CryptographicSponge,
    PCS: PolynomialCommitment<F, Poly<F>, S>,
>(node: impl NodeOpsNative<i8> + 'static) -> Box<dyn NodeOpsProve<F, S, PCS, i8>> {
    let node = &node as &dyn Any;

    match node.downcast_ref::<ReLUNode<i8>>() {
        Some(s) => return Box::new((*s).clone()),
        _ => {}
    }

    match node.downcast_ref::<ReshapeNode>() {
        Some(s) => return Box::new((*s).clone()),
        _ => {}
    }

    panic!("No implementor of NodeOpsProve was received");
}


#[test]
fn test_downcast_relu() {
    let node1 = ReLUNode::new(1, 1);
    let node2 = ReshapeNode::new(vec![1, 2], vec![1, 2]);

    let node1_proof = node_downcast::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(node1);
    let node2_proof = node_downcast::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(node2);

    let nodes: Vec<Box<dyn NodeOpsNative<i8>>> = vec![Box::new(node1), Box::new(node2)];
    let nodes_proof: Vec<Box<dyn NodeOpsProve<Fr, PoseidonSponge<Fr>, Ligero<Fr>, i8>>> = nodes
    .into_iter()
    .map(|x| node_downcast::<Fr, PoseidonSponge<Fr>, Ligero<Fr>>(*(x.as_ref())))
    .collect();

    println!("Node 1: {}", node1_proof.type_name());
    println!("Node 2: {}", node2_proof.type_name());

}