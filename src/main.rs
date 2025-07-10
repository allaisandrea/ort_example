fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).unwrap_or_else(|| {
        eprintln!("Usage: {} <path_to_onnx_file>", args[0]);
        std::process::exit(1);
    });
    let execution_provider = ort::execution_providers::CUDAExecutionProvider::default().build();
    let mut session = ort::session::Session::builder()?
        .with_execution_providers([execution_provider])?
        .commit_from_file(path)?;
    let memory_info = ort::memory::MemoryInfo::new(
        ort::memory::AllocationDevice::CPU,
        0,
        ort::memory::AllocatorType::Device,
        ort::memory::MemoryType::Default,
    )?;
    let mut binding = session.create_binding()?;
    let buffer_in = ort::value::Tensor::<f32>::from_array(([3], vec![0.1, 0.2, 0.3]))?;
    binding.bind_input("tensor".to_string(), &buffer_in)?;
    binding.bind_output_to_device("tensor".to_string(), &memory_info)?;
    let mut outputs = session.run_binding(&binding)?;
    let output_tensor: ort::value::Tensor<f32> = outputs.remove("tensor").unwrap().downcast()?;

    let execution_provider = ort::execution_providers::CUDAExecutionProvider::default().build();
    let mut session1 = ort::session::Session::builder()?
        .with_execution_providers([execution_provider])?
        .commit_from_file(path)?;
    let memory_info = ort::memory::MemoryInfo::new(
        ort::memory::AllocationDevice::CPU,
        0,
        ort::memory::AllocatorType::Device,
        ort::memory::MemoryType::Default,
    )?;
    let mut binding = session1.create_binding()?;
    binding.bind_input("tensor".to_string(), &output_tensor)?;
    binding.bind_output_to_device("tensor".to_string(), &memory_info)?;
    let mut outputs = session1.run_binding(&binding)?;
    let output_tensor: ort::value::Tensor<f32> = outputs.remove("tensor").unwrap().downcast()?;
    let (shape, data) = output_tensor.extract_tensor();
    println!("Shape: {:?}", shape);
    println!("Data: {:?}", data);
    Ok(())
}
