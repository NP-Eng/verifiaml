macro_rules! node_operation {
    ($self:expr, $method:ident, $($arg:expr),*) => {
        match $self {
            Node::BMM(node) => node.$method($($arg),*),
            Node::RequantiseBMMFloat(node) => node.$method($($arg),*),
            // TODO add Node::RequantiseBMMRef(node) => node.$method($($arg),*), once the latter implements commit, proof
            Node::RequantiseBMMRef(_) => unimplemented!(),
            Node::ReLU(node) => node.$method($($arg),*),
            Node::Reshape(node) => NodeOpsProve::<_, _, _, I, _>::$method(node, $($arg),*),
        }
    };
}
