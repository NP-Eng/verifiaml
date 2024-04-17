macro_rules! node_operation {
    ($self:expr, $method:ident, $($arg:expr),*) => {
        match $self {
            Node::BMM(node) => node.$method($($arg),*),
            Node::RequantizeBMMFloat(node) => node.$method($($arg),*),
            // TODO add Node::RequantizeBMMRef(node) => node.$method($($arg),*), once the latter implements commit, proof
            Node::RequantizeBMMRef(_) => unimplemented!(),
            Node::RequantizeBMMSingle(_) => unimplemented!(),
            Node::ReLU(node) => node.$method($($arg),*),
            Node::Reshape(node) => NodeOpsProve::<_, _, _, I, _>::$method(node, $($arg),*),
        }
    };
}
