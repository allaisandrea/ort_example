fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: {} <path_to_onnx_file>", args[0]);
        std::process::exit(1);
    });
    let execution_provider = ort::execution_providers::CUDAExecutionProvider::default().build();
    let _session = ort::session::Session::builder()?
        .with_execution_providers([execution_provider])?
        .commit_from_file(path)?;
    Ok(())
}
