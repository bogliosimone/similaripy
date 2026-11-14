//! gRPC code from proto/ (server plan S4 P1): protox compiles the .proto in
//! Rust, so no protoc is needed on dev machines or CI.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=proto");
    let fds = protox::compile(["similaripy/v1/recommender.proto"], ["proto"])?;
    tonic_prost_build::configure().compile_fds(fds)?;
    Ok(())
}
