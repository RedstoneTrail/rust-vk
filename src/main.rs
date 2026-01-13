use std::cmp::min;
use std::sync::Arc;

use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::{
    StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo, PrimaryAutoCommandBuffer,
    RenderPassBeginInfo, SubpassBeginInfo, SubpassContents, SubpassEndInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::image::sampler::{Sampler, SamplerCreateInfo};
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::instance::{Instance, InstanceCreateFlags, InstanceCreateInfo};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    GraphicsPipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::shader::ShaderModule;
use vulkano::swapchain;
use vulkano::swapchain::{Surface, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo};
use vulkano::sync::future::FenceSignalFuture;
use vulkano::sync::{self, GpuFuture};
use vulkano::{Validated, VulkanError, VulkanLibrary};

use winit::event::{DeviceEvent, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;

#[derive(BufferContents, Vertex, Clone, Copy, Default)]
#[repr(C)]
struct Point {
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
    #[format(R32G32_SFLOAT)]
    uv: [f32; 2],
    #[format(R32G32B32_SFLOAT)]
    normal: [f32; 3],
    #[format(R32G32B32_SFLOAT)]
    color: [f32; 3],
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/vertex.vert",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/fragment.frag",
    }
}

fn get_pipeline(
    device: Arc<Device>,
    vs: Arc<ShaderModule>,
    fs: Arc<ShaderModule>,
    render_pass: Arc<RenderPass>,
    viewport: Viewport,
) -> Arc<GraphicsPipeline> {
    let vs = vs.entry_point("main").unwrap();
    let fs = fs.entry_point("main").unwrap();

    let vertex_input_state = Point::per_vertex()
        .definition(&vs.info().input_interface)
        .unwrap();

    let stages = [
        PipelineShaderStageCreateInfo::new(vs),
        PipelineShaderStageCreateInfo::new(fs),
    ];

    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )
    .unwrap();

    let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

    GraphicsPipeline::new(
        device.clone(),
        None,
        GraphicsPipelineCreateInfo {
            stages: stages.into_iter().collect(),
            vertex_input_state: Some(vertex_input_state),
            input_assembly_state: Some(InputAssemblyState::default()),
            viewport_state: Some(ViewportState {
                viewports: [viewport].into_iter().collect(),
                ..Default::default()
            }),
            rasterization_state: Some(RasterizationState::default()),
            multisample_state: Some(MultisampleState::default()),
            color_blend_state: Some(ColorBlendState::with_attachment_states(
                subpass.num_color_attachments(),
                ColorBlendAttachmentState::default(),
            )),
            subpass: Some(subpass.into()),
            ..GraphicsPipelineCreateInfo::layout(layout)
        },
    )
    .unwrap()
}

fn get_command_buffers(
    command_buffer_allocator: &StandardCommandBufferAllocator,
    device: Arc<Device>,
    queue: &Arc<Queue>,
    pipeline: &Arc<GraphicsPipeline>,
    framebuffers: &Vec<Arc<Framebuffer>>,
    vertex_buffer: Subbuffer<[Point]>,
    index_buffer: &Subbuffer<[u32]>,
    push_constants: vs::PushConstants,
    sampler: &Arc<Sampler>,
    texture: &Arc<ImageView>,
    frame_idx: u32,
) -> Vec<Arc<PrimaryAutoCommandBuffer>> {
    let descriptor_set_allocator =
        StandardDescriptorSetAllocator::new(device.clone(), Default::default());

    let layouts = pipeline.layout().set_layouts();
    println!("\tlisting {:?} descriptor set layouts", layouts.len());
    for layout in layouts.into_iter() {
        dbg!(layout);
    }
    let layout = layouts.get(frame_idx as usize).unwrap();
    let set = PersistentDescriptorSet::new(
        &descriptor_set_allocator,
        layout.clone(),
        [WriteDescriptorSet::image_view_sampler(
            0,
            texture.clone(),
            sampler.clone(),
        )],
        [],
    )
    .unwrap();

    framebuffers
        .iter()
        .map(|framebuffer| {
            let mut builder = AutoCommandBufferBuilder::primary(
                command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::MultipleSubmit,
            )
            .unwrap();

            builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values: vec![Some([0.1, 0.1, 0.1, 1.0].into())],
                        ..RenderPassBeginInfo::framebuffer(framebuffer.clone())
                    },
                    SubpassBeginInfo {
                        contents: SubpassContents::Inline,
                        ..Default::default()
                    },
                )
                .unwrap()
                .bind_pipeline_graphics(pipeline.clone())
                .unwrap()
                .push_constants(pipeline.layout().clone(), 0, push_constants)
                .unwrap()
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .unwrap()
                .bind_index_buffer(index_buffer.clone())
                .unwrap()
                .bind_descriptor_sets(
                    vulkano::pipeline::PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
                )
                .unwrap()
                .draw_indexed(index_buffer.len() as u32, 1, 0, 0, 0)
                .unwrap()
                .end_render_pass(SubpassEndInfo::default())
                .unwrap();

            builder.build().unwrap()
        })
        .collect()
}

fn get_render_pass(
    device: Arc<Device>,
    swapchain: &Arc<Swapchain>,
) -> Arc<vulkano::render_pass::RenderPass> {
    return vulkano::single_pass_renderpass!(
        device,
        attachments: {
            color: {
                format: swapchain.image_format(),
                samples: 1,
                load_op: Clear,
                store_op: Store,
            },
        },
        pass: {
            color: [color],
            depth_stencil: {},
        },
    )
    .expect("render pass creation failed");
}

fn select_physical_device(
    instance: &Arc<Instance>,
    surface: &Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("cant enumerate devices")
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                // pick the first queue family works
                .position(|(i, q)| {
                    q.queue_flags.contains(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|q| (p, q as u32))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            _ => 4,
        })
        .expect("no suitable device available")
}

fn create_instance(
    library: &Arc<VulkanLibrary>,
    event_loop: &EventLoop<()>,
) -> Result<Arc<Instance>, Validated<VulkanError>> {
    return Instance::new(
        library.clone(),
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: Surface::required_extensions(event_loop),
            ..Default::default()
        },
    );
}

fn get_framebuffers(images: &[Arc<Image>], render_pass: Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

fn main() {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");
    let event_loop = EventLoop::new();

    let instance = create_instance(&library, &event_loop).expect("failed to create instance");

    println!("instance made");

    let window = Arc::new(WindowBuilder::new().build(&event_loop).unwrap());

    let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, _) = select_physical_device(&instance, &surface, &device_extensions);

    println!("device made");

    for family in physical_device.queue_family_properties() {
        println!("family found with {:?} queues", family.queue_count);
    }

    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .position(|queue_family_properties| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("failed to find suitable queue") as u32;

    println!("found queue");

    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            enabled_extensions: device_extensions,
            ..Default::default()
        },
    )
    .expect("failed to make device");

    println!("made device");

    let queue = queues.next().unwrap();

    println!("picked queue");

    let caps = physical_device
        .surface_capabilities(&surface, Default::default())
        .expect("surface capabilities get failed");

    println!("surface capabilities got");

    let dimensions = window.inner_size();
    let composite_alpha = caps.supported_composite_alpha.into_iter().next().unwrap();
    let image_format = physical_device
        .surface_formats(&surface, Default::default())
        .unwrap()[0]
        .0;

    let (mut swapchain, images) = Swapchain::new(
        device.clone(),
        surface.clone(),
        SwapchainCreateInfo {
            min_image_count: caps.min_image_count + 1,
            image_format,
            image_extent: dimensions.into(),
            image_usage: ImageUsage::COLOR_ATTACHMENT,
            composite_alpha,
            ..Default::default()
        },
    )
    .expect("swapchain creation failed");

    println!(
        "swapchain created with {:?} images",
        caps.min_image_count + 1
    );

    let render_pass = get_render_pass(device.clone(), &swapchain);

    println!("render pass created");

    let mut framebuffers = get_framebuffers(&images, render_pass.clone());

    println!("framebuffers created");

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let verts = [
        Point {
            position: [0.5, 0.5, 0.5],
            uv: [1.0, 1.0],
            ..Default::default()
        },
        Point {
            position: [-0.5, 0.5, 0.5],
            uv: [-1.0, 1.0],
            ..Default::default()
        },
        Point {
            position: [0.5, -0.5, 0.5],
            uv: [1.0, -1.0],
            ..Default::default()
        },
        Point {
            position: [-0.5, -0.5, 0.5],
            uv: [-1.0, -1.0],
            ..Default::default()
        },
        Point {
            position: [0.5, 0.5, -0.5],
            uv: [1.0, 1.0],
            ..Default::default()
        },
        Point {
            position: [-0.5, 0.5, -0.5],
            uv: [-1.0, 1.0],
            ..Default::default()
        },
        Point {
            position: [0.5, -0.5, -0.5],
            uv: [1.0, -1.0],
            ..Default::default()
        },
        Point {
            position: [-0.5, -0.5, -0.5],
            uv: [-1.0, -1.0],
            ..Default::default()
        },
    ];

    let vertex_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::VERTEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        verts.into_iter(),
    )
    .unwrap();

    println!("vertex buffer created");

    let mut res = min(window.inner_size().height, window.inner_size().width) as f32;

    let mut push_constants = vs::PushConstants {
        origin: [0.0, 0.0, 0.3].into(),
        rotation: [0.0, 0.0].into(),
        resolution: window.inner_size().into(),
        side_length: res,
    };

    let vs = vs::load(device.clone()).expect("vertex shader module creation failed");
    let fs = fs::load(device.clone()).expect("fragment shader module creation failed");

    println!("shaders loaded");

    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [res, res],
        depth_range: 0.0..=1.0,
    };

    let pipeline = get_pipeline(
        device.clone(),
        vs.clone(),
        fs.clone(),
        render_pass.clone(),
        viewport.clone(),
    );

    println!("pipeline created");

    let command_buffer_allocator = StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    );

    let index_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::INDEX_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        [0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5].into_iter(),
    )
    .unwrap();

    println!("index buffer created");

    let texture = image::load_from_memory(include_bytes!("../texture.png")).unwrap();

    println!(
        "loaded image dimensions: {:?}x{:?}",
        texture.width(),
        texture.height()
    );

    let image: Arc<Image>;

    let texture = {
        let mut uploads = AutoCommandBufferBuilder::primary(
            &command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let staging_buffer = Buffer::new_slice(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (texture.width() * texture.height() * 3) as u64,
        )
        .unwrap();

        let image_data = &mut *staging_buffer.write().unwrap();

        let bytes = texture.as_bytes();

        println!(
            "texture.as_bytes().len() = {:?}\ntexture.width() * texture.height() * 3 = {:?}",
            texture.as_bytes().len(),
            texture.width() * texture.height() * 3
        );

        for i in 0..bytes.len() - 1 {
            image_data[i] = bytes[i].clone();
        }

        image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::R8G8B8_SRGB,
                extent: [texture.width(), texture.height(), 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                sharing: sync::Sharing::Exclusive,
                tiling: vulkano::image::ImageTiling::Optimal,
                samples: vulkano::image::SampleCount::Sample1,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        uploads
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging_buffer.clone(),
                image.clone(),
            ))
            .unwrap();

        dbg!();

        let image = Image::new(
            memory_allocator.clone(),
            ImageCreateInfo {
                image_type: ImageType::Dim2d,
                format: vulkano::format::Format::R8G8B8_SRGB,
                extent: [texture.width(), texture.height(), 1],
                usage: ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED,
                sharing: sync::Sharing::Exclusive,
                tiling: vulkano::image::ImageTiling::Optimal,
                samples: vulkano::image::SampleCount::Sample1,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .unwrap();

        ImageView::new_default(image).unwrap()
    };

    let sampler = Sampler::new(device.clone(), SamplerCreateInfo::simple_repeat_linear()).unwrap();

    println!("texture created");

    let mut command_buffers = get_command_buffers(
        &command_buffer_allocator,
        device.clone(),
        &queue,
        &pipeline,
        &framebuffers.clone(),
        vertex_buffer.clone(),
        &index_buffer,
        push_constants.clone(),
        &sampler,
        &texture,
        0,
    );

    println!("initial command buffers created");

    let frames_in_flight = images.len();
    let mut fences: Vec<Option<Arc<FenceSignalFuture<_>>>> = vec![None; frames_in_flight];
    let mut previous_fence_i = 0;

    let mut window_resized = false;
    let mut recreate_swapchain = false;
    let mut frame_idx: u32 = 0;

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent { event: ev, .. } => match ev {
            WindowEvent::CloseRequested => {
                println!("window close requested");
                *control_flow = ControlFlow::Exit;
            }
            WindowEvent::Resized(_) => {
                window_resized = true;
                println!("window resized to {:?}", window.inner_size());
            }
            WindowEvent::KeyboardInput { input, .. } => {
                // only quit on release of Q (no auto repeat)
                if input.virtual_keycode.is_some() {
                    if input.state == winit::event::ElementState::Released
                        && input.virtual_keycode.unwrap() == VirtualKeyCode::Q
                    {
                        println!("Q was pressed, quitting");
                        *control_flow = ControlFlow::Exit;
                    }
                }

                // only accept input when it is pressed (gives auto repeat)
                if input.virtual_keycode.is_some()
                    && input.state == winit::event::ElementState::Pressed
                {
                    match input.virtual_keycode.unwrap() {
                        VirtualKeyCode::K => {
                            println!("K was pressed, moving away");

                            push_constants.origin[2] -= <f32 as Into<f32>>::into(0.1);
                        }
                        VirtualKeyCode::J => {
                            println!("J was pressed, moving toward");

                            push_constants.origin[2] += <f32 as Into<f32>>::into(0.1);
                        }
                        VirtualKeyCode::H => {
                            println!("H was pressed, moving left");

                            push_constants.origin[0] -= <f32 as Into<f32>>::into(0.1);
                        }
                        VirtualKeyCode::L => {
                            println!("L was pressed, moving right");

                            push_constants.origin[0] += <f32 as Into<f32>>::into(0.1);
                        }
                        _ => {
                            // println!("key pressed");
                        }
                    }
                }
            }
            _ => {}
        },
        Event::DeviceEvent { event: ev, .. } => match ev {
            DeviceEvent::MouseMotion { delta } => {
                // println!("mouse moved by {:?}", delta);

                push_constants.rotation[0] +=
                    <f32 as Into<f32>>::into(delta.0 as f32 / push_constants.resolution[0]);
                push_constants.rotation[1] +=
                    <f32 as Into<f32>>::into(delta.1 as f32 / push_constants.resolution[1]);

                // println!("rotation is now {:?}", push_constants.rotation);
            }
            _ => {
                // println!("device event recieved: ev={:?} from id={:?}", ev, id);
            }
        },
        Event::UserEvent(_) => {
            println!("user event recieved");
        }
        Event::MainEventsCleared => {
            if window_resized || recreate_swapchain {
                recreate_swapchain = false;

                let new_dimensions = window.inner_size();

                res = min(new_dimensions.width, new_dimensions.height) as f32;

                let (new_swapchain, new_images) = swapchain
                    .recreate(SwapchainCreateInfo {
                        image_extent: new_dimensions.into(),
                        ..swapchain.create_info()
                    })
                    .expect("swapchain recreation failed because: {e}");

                swapchain = new_swapchain;
                let new_framebuffers = get_framebuffers(&new_images, render_pass.clone());

                if window_resized {
                    window_resized = false;

                    viewport.extent = [res, res];

                    let new_pipeline = get_pipeline(
                        device.clone(),
                        vs.clone(),
                        fs.clone(),
                        render_pass.clone(),
                        viewport.clone(),
                    );

                    command_buffers = get_command_buffers(
                        &command_buffer_allocator,
                        device.clone(),
                        &queue,
                        &new_pipeline,
                        &new_framebuffers.clone(),
                        vertex_buffer.clone(),
                        &index_buffer,
                        push_constants.clone(),
                        &sampler,
                        &texture,
                        frame_idx,
                    );

                    framebuffers = new_framebuffers;
                }
            } else {
                let new_dimensions = window.inner_size();
                viewport.extent = new_dimensions.into();
                push_constants.resolution = new_dimensions.into();
                push_constants.side_length = res;

                let new_pipeline = get_pipeline(
                    device.clone(),
                    vs.clone(),
                    fs.clone(),
                    render_pass.clone(),
                    viewport.clone(),
                );

                frame_idx += 1;

                command_buffers = get_command_buffers(
                    &command_buffer_allocator,
                    device.clone(),
                    &queue,
                    &new_pipeline,
                    &framebuffers.clone(),
                    vertex_buffer.clone(),
                    &index_buffer,
                    push_constants.clone(),
                    &sampler,
                    &texture,
                    frame_idx,
                );
            }

            let (image_i, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None)
                    .map_err(Validated::unwrap)
                {
                    Ok(r) => r,
                    Err(VulkanError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("next image acquisition failed: {e}"),
                };

            // swapchain works, but will not be properly displayed, so regenerate it
            if suboptimal {
                recreate_swapchain = true;
            }

            // wait until the fence for this image is finished
            if let Some(image_fence) = &fences[image_i as usize] {
                image_fence
                    .wait(None)
                    .expect("failed to wait for next image");
            }

            // create a previous future if it doesnt already exist
            let previous_future = match fences[previous_fence_i as usize].clone() {
                // no future exists, make a NowFuture
                None => {
                    let mut now = sync::now(device.clone());
                    now.cleanup_finished();
                    now.boxed()
                }
                // FenceSignalFuture exists, use it
                Some(fence) => fence.boxed(),
            };

            let future = previous_future
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffers[image_i as usize].clone())
                .expect("execution failed")
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_i),
                )
                .then_signal_fence_and_flush();

            match future {
                Err(e) => {
                    dbg!(&e);
                    panic!("fence validation failed: {e}");
                }
                _ => {
                    fences[image_i as usize] = match future.map_err(Validated::unwrap) {
                        Ok(value) => Some(Arc::new(value)),
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            None
                        }
                        Err(e) => {
                            println!("future flush failed: {e}");
                            None
                        }
                    };
                }
            };

            previous_fence_i = image_i;
        }
        _ => (),
    });
}
