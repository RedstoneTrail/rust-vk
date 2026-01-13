use image::DynamicImage;
use std::{fmt::Debug, sync::Arc};
use vulkano::{
    DeviceSize, VulkanLibrary,
    buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferToImageInfo,
        PrimaryAutoCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
        SubpassEndInfo,
        allocator::{CommandBufferAllocator, StandardCommandBufferAllocator},
    },
    device::{
        Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
        physical::{PhysicalDevice, PhysicalDeviceType},
    },
    format::Format,
    image::{Image, ImageCreateInfo, ImageFormatInfo, ImageUsage, view::ImageView},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        graphics::{
            GraphicsPipelineCreateInfo,
            color_blend::{ColorBlendAttachmentState, ColorBlendState},
            input_assembly::InputAssemblyState,
            multisample::MultisampleState,
            rasterization::RasterizationState,
            vertex_input::{Vertex, VertexDefinition},
            viewport::{Viewport, ViewportState},
        },
        layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderModule,
    swapchain::{Surface, Swapchain, SwapchainCreateInfo},
};
use winit::{
    application::ApplicationHandler,
    event::{StartCause, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

mod vs {
    vulkano_shaders::shader!(
        ty: "vertex",
        path: "src/vertex.vert"
    );
}

mod fs {
    vulkano_shaders::shader!(
        ty: "fragment",
        path: "src/fragment.frag"
    );
}

static DEBUG_PADDING: usize = 16;

mod debug_names {
    pub const WINDOW_MANAGEMENT: &'static str = "WINDOW_MGMT";
    pub const VULKANO_INITIALISATION: &'static str = "VULKANO_INIT";
    pub const IMAGE_MANAGEMENT: &'static str = "IMAGE_MGMT";
    pub const VERTEX_MANAGEMENT: &'static str = "VERTEX_MGMT";
    pub const INDEX_MANAGEMENT: &'static str = "INDEX_MGMT";
    pub const COMMAND_BUFFER_CREATION: &'static str = "CMD_BUF_CREATION";
    pub const GENERAL: &'static str = "GENERAL";
}

macro_rules! debug_message {
    ($context:expr, $content:expr) => {
        if cfg!(debug_assertions) {
            eprintln!("[{:<padding$}]: {}", $context, $content, padding=DEBUG_PADDING);
        }
    };
    ($context:expr, $format:expr, $($values:expr),+) => {
        if cfg!(debug_assertions) {
            eprintln!("[{:<padding$}]: {}", $context, format!($format, $($values),+), padding=DEBUG_PADDING);
        }
    };
}

fn hash_string(string: &'static str) -> usize {
    let mut hash: usize = 0;
    for character in string.bytes() {
        hash += hash.checked_shl(5).unwrap_or(0) + character as usize;
    }
    return hash;
}

#[derive(BufferContents, Vertex, Clone, Copy)]
#[repr(C)]
struct Point {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

impl Debug for Point {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({},{})", self.position[0], self.position[1])
    }
}

#[derive(Clone, Debug)]
struct IndexSet {
    indices: Vec<u32>,
    // label: String,
    label: usize,
}

#[derive(Clone)]
#[allow(unused)]
struct VulkanoInstance {
    library: Arc<VulkanLibrary>,
    vk_instance: Arc<Instance>,
    physical_device: Arc<PhysicalDevice>,
    logical_device: Arc<Device>,
    queue_family_index: u32,
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<dyn CommandBufferAllocator>,
    attached_images: Vec<Arc<Image>>,
    vertex_buffer: Option<Subbuffer<[Point]>>,
    verticies: Vec<Point>,
    index_buffer: Option<Subbuffer<[u32]>>,
    index_sets: Vec<IndexSet>,
    vertex_shader: Arc<ShaderModule>,
    fragment_shader: Arc<ShaderModule>,
    command_buffers: Option<Vec<Arc<PrimaryAutoCommandBuffer>>>,
    framebuffers: Vec<Arc<Framebuffer>>,
    pipeline: Arc<GraphicsPipeline>,
    render_pass: Arc<RenderPass>,
    surface: Arc<Surface>,
}

impl VulkanoInstance {
    fn new(
        window: Arc<Window>,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) -> VulkanoInstance {
        debug_message!(debug_names::VULKANO_INITIALISATION, "initialising vulkano");

        let library = VulkanLibrary::new().expect("vulkan library");
        let required_extensions =
            Surface::required_extensions(event_loop).expect("required extensions");

        debug_message!(
            debug_names::VULKANO_INITIALISATION,
            "enabling extensions: {:?}",
            required_extensions
        );

        let vk_instance = Instance::new(
            library.clone(),
            InstanceCreateInfo {
                flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .expect("instance creation");

        let surface =
            Surface::from_window(vk_instance.clone(), window.clone()).expect("surface creation");

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..Default::default()
        };

        let (physical_device, queue_family_index) = vk_instance
            .enumerate_physical_devices()
            .expect("enumeration of physical devices")
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
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
            .expect("available device");

        for family in physical_device.queue_family_properties() {
            debug_message!(
                debug_names::VULKANO_INITIALISATION,
                "chosen physical device has a family with {:?} queue(s)",
                family.queue_count
            );
        }

        let (logical_device, mut queues) = Device::new(
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
        .expect("device creation");

        debug_message!(
            debug_names::VULKANO_INITIALISATION,
            "logical device created with extensions: {:?}",
            device_extensions
        );

        let queue = queues.next().expect("a queue");

        let capabilities = physical_device
            .surface_capabilities(&surface, Default::default())
            .expect("get surface capabilities");

        let memory_allocator =
            Arc::new(StandardMemoryAllocator::new_default(logical_device.clone()));

        let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
            logical_device.clone(),
            Default::default(),
        ));

        let viewport_extent = [
            window.inner_size().width as f32,
            window.inner_size().height as f32,
        ];

        let composite_alpha = capabilities
            .supported_composite_alpha
            .into_iter()
            .next()
            .expect("composite alpha");
        let image_format = physical_device
            .surface_formats(&surface, Default::default())
            .expect("surface formats")[0]
            .0;

        let viewport = Viewport {
            offset: [0.0, 0.0],
            extent: viewport_extent,
            depth_range: 0.0..=1.0,
        };

        let (mut swapchain, images) = Swapchain::new(
            logical_device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: capabilities.min_image_count + 1,
                image_format,
                composite_alpha,
                image_extent: [viewport_extent[0] as u32, viewport_extent[1] as u32],
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                ..Default::default()
            },
        )
        .expect("swapchain");

        let render_pass = vulkano::single_pass_renderpass!(
            logical_device.clone(),
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
        .expect("render pass");

        let framebuffers = get_framebuffers(&images, &render_pass);

        debug_message!(
            debug_names::VULKANO_INITIALISATION,
            "creating viewport with extent {:?}",
            viewport_extent
        );

        let vs = vs::load(logical_device.clone()).expect("load vertex shader");
        let fs = fs::load(logical_device.clone()).expect("load fragment shader");

        let pipeline = {
            let vs = vs.entry_point("main").expect("vertex shader with main()");
            let fs = fs.entry_point("main").expect("fragment shader with main()");

            let vertex_input_state = Point::per_vertex()
                .definition(&vs)
                .expect("vertex input state");

            let stages = [
                PipelineShaderStageCreateInfo::new(vs),
                PipelineShaderStageCreateInfo::new(fs),
            ];

            let layout = PipelineLayout::new(
                logical_device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(logical_device.clone())
                    .expect("pipeline layout create info"),
            )
            .expect("pipeline layout");

            let subpass = Subpass::from(render_pass.clone(), 0).expect("subpass");

            GraphicsPipeline::new(
                logical_device.clone(),
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
            .expect("graphics pipeline")
        };

        debug_message!(debug_names::VULKANO_INITIALISATION, "instance initialised");

        return VulkanoInstance {
            library,
            vk_instance,
            physical_device,
            queue_family_index,
            queue,
            memory_allocator,
            command_buffer_allocator,
            attached_images: Vec::new(),
            vertex_buffer: None,
            verticies: Vec::new(),
            index_buffer: None,
            index_sets: Vec::new(),
            vertex_shader: vs,
            fragment_shader: fs,
            command_buffers: None,
            framebuffers,
            pipeline,
            logical_device,
            render_pass,
            surface,
        };
    }

    fn get_command_buffers(&mut self) {
        debug_message!(
            debug_names::COMMAND_BUFFER_CREATION,
            "generating new command buffers"
        );

        self.command_buffers = Some(
            self.framebuffers
                .iter()
                .map(|framebuffer| {
                    let mut builder = AutoCommandBufferBuilder::primary(
                        self.command_buffer_allocator.clone(),
                        self.queue_family_index,
                        CommandBufferUsage::MultipleSubmit,
                    )
                    .unwrap();

                    unsafe {
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
                            .bind_pipeline_graphics(self.pipeline.clone())
                            .unwrap()
                            .bind_vertex_buffers(0, self.vertex_buffer.clone().unwrap().clone())
                            .unwrap()
                            .bind_index_buffer(self.index_buffer.clone().unwrap().clone())
                            .unwrap()
                            .draw_indexed(
                                self.index_buffer.clone().unwrap().len() as u32,
                                1,
                                0,
                                0,
                                0,
                            )
                            .unwrap()
                            .end_render_pass(SubpassEndInfo::default())
                            .unwrap();
                    }

                    builder.build().unwrap()
                })
                .collect(),
        );

        debug_message!(
            debug_names::COMMAND_BUFFER_CREATION,
            "new command buffers generated"
        );
    }

    fn _create_image(&mut self, image_create_info: ImageCreateInfo) -> usize {
        let image = Image::new(
            self.memory_allocator.clone(),
            image_create_info,
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("image creation");

        let dimensions = image.extent();

        debug_message!(
            debug_names::IMAGE_MANAGEMENT,
            "created new image\n\tsize: {:?}x{:?}x{:?}",
            dimensions[0],
            dimensions[1],
            dimensions[0]
        );

        let image_idx = self.attached_images.len();

        self.attached_images.push(image);

        return image_idx;
    }

    fn image_from_data(&mut self, data: DynamicImage) -> usize {
        debug_message!(debug_names::IMAGE_MANAGEMENT, "uploading a new image");

        let mut staging = AutoCommandBufferBuilder::primary(
            self.command_buffer_allocator.clone(),
            self.queue_family_index,
            vulkano::command_buffer::CommandBufferUsage::OneTimeSubmit,
        )
        .expect("staging command buffer");

        let (format, extent, _channels) = match data.clone() {
            DynamicImage::ImageRgba8(img) => (
                Format::R8G8B8A8_SRGB,
                [img.dimensions().0, img.dimensions().0, 1],
                4,
            ),
            DynamicImage::ImageRgb8(img) => (
                Format::R8G8B8_SRGB,
                [img.dimensions().0, img.dimensions().0, 1],
                3,
            ),
            e => {
                debug_message!(
                    debug_names::IMAGE_MANAGEMENT,
                    "unexpected image format: {:?}",
                    e
                );
                panic!();
            }
        };

        let usage = ImageUsage::TRANSFER_DST | ImageUsage::SAMPLED;
        let image_type = vulkano::image::ImageType::Dim2d;
        let format_info = ImageFormatInfo {
            image_type,
            usage,
            format,
            ..Default::default()
        };

        match self
            .physical_device
            .image_format_properties(format_info.clone())
        {
            Ok(None) => {
                debug_message!(
                    debug_names::IMAGE_MANAGEMENT,
                    "unsupported image format: {:?}",
                    format_info
                );

                panic!();
            }
            Ok(_) => {
                debug_message!(debug_names::IMAGE_MANAGEMENT, "image format is supported");
            }
            Err(_) => {}
        }

        let image = Image::new(
            self.memory_allocator.clone(),
            ImageCreateInfo {
                image_type,
                usage,
                format,
                extent,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
        )
        .expect("texture image creation");

        let staging_buffer = Buffer::new_slice(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::TRANSFER_SRC,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_HOST
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            (data.clone().width() * data.clone().height() * 4) as DeviceSize,
        )
        .expect("staging buffer creation");

        let staging_buffer_clone = staging_buffer.clone();
        let mut buffer_write = staging_buffer_clone.write().expect("read staging buffer");

        (0..buffer_write.len()).into_iter().for_each(|idx| {
            buffer_write[idx] = data.as_bytes().to_vec()[idx / 4];
        });

        staging
            .copy_buffer_to_image(CopyBufferToImageInfo::buffer_image(
                staging_buffer.clone(),
                image.clone(),
            ))
            .unwrap();

        debug_message!(debug_names::IMAGE_MANAGEMENT, "uploaded image");

        let image_idx = self.attached_images.len();

        self.attached_images.push(image);

        return image_idx;
    }

    fn add_verticies(&mut self, verticies: Vec<Point>) -> Vec<u32> {
        let mut indices = Vec::new();

        debug_message!(
            debug_names::VERTEX_MANAGEMENT,
            "old vertex list: {:?}",
            self.verticies.clone()
        );
        debug_message!(
            debug_names::VERTEX_MANAGEMENT,
            "adding vertices: {:?}",
            verticies.clone()
        );

        for vertex in verticies {
            indices.push(self.verticies.len() as u32);
            self.verticies.push(vertex);
        }

        let vertex_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            self.verticies.clone(),
        )
        .expect("vertex buffer creation");

        self.vertex_buffer = Some(vertex_buffer);

        debug_message!(
            debug_names::VERTEX_MANAGEMENT,
            "new vertex list: {:?}",
            self.verticies.clone()
        );
        debug_message!(
            debug_names::VERTEX_MANAGEMENT,
            "vertices uploaded at indices: {:?}",
            indices
        );

        return indices;
    }

    fn add_index_set(&mut self, index_set: IndexSet) {
        for existing_index_set in self.index_sets.clone() {
            if existing_index_set.label == index_set.label {
                panic!("tried to make a new index set with an existing set's name");
            }
        }

        self.index_sets.push(index_set.clone());

        debug_message!(
            debug_names::INDEX_MANAGEMENT,
            "creating new index buffer, adding indices: {:?} for thing labelled {}",
            index_set.clone().indices,
            index_set.clone().label
        );

        let index_buffer = Buffer::from_iter(
            self.memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::HOST_SEQUENTIAL_WRITE
                    | MemoryTypeFilter::PREFER_DEVICE,
                ..Default::default()
            },
            index_set.clone().indices,
        )
        .expect("index buffer creation");

        self.index_buffer = Some(index_buffer);
    }

    // fn _remove_index_set(&mut self, label: String) {
    fn _remove_index_set(&mut self, label: usize) {
        for i in 0..self.index_sets.len() {
            if self.index_sets[i].label == label {
                debug_message!(
                    debug_names::INDEX_MANAGEMENT,
                    "removing index set {} with label {}",
                    i,
                    label
                );
                self.index_sets.remove(i);
            }
        }
    }
}

fn get_framebuffers(images: &[Arc<Image>], render_pass: &Arc<RenderPass>) -> Vec<Arc<Framebuffer>> {
    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).expect("image view");
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view],
                    ..Default::default()
                },
            )
            .expect("framebuffer")
        })
        .collect::<Vec<_>>()
}

#[allow(unused)]
struct InitialisedApp {
    window: Arc<Window>,
    vk_instance: VulkanoInstance,
    texture_idx: usize,
}

#[derive(Default)]
enum App {
    #[default]
    Uninitialised,
    Initialised(InitialisedApp),
}

impl ApplicationHandler for App {
    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        debug_message!(debug_names::WINDOW_MANAGEMENT, "window resuming");
    }

    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        match cause {
            StartCause::Init => {
                let window = Arc::new(
                    event_loop
                        .create_window(Window::default_attributes())
                        .expect("window creation"),
                );

                let mut vk_instance = VulkanoInstance::new(window.clone(), event_loop);

                debug_message!(debug_names::WINDOW_MANAGEMENT, "created window and surface");

                let data = image::load_from_memory(include_bytes!("../texture.png")).unwrap();

                let texture_idx = vk_instance.image_from_data(data);
                let index_list = vk_instance.add_verticies(vec![
                    Point {
                        position: [0.5, 0.5],
                    },
                    Point {
                        position: [0.5, -0.5],
                    },
                    Point {
                        position: [-0.5, 0.5],
                    },
                    Point {
                        position: [-0.5, -0.5],
                    },
                ]);

                let index_set = IndexSet {
                    indices: index_list,
                    label: hash_string("The Shape"),
                    // label: "The Shape".to_string(),
                };

                vk_instance.add_index_set(index_set);
                // self.vk_instance
                //     .as_mut()
                //     .unwrap()
                //     .remove_index_set("The Shape".to_string());

                *self = App::Initialised(InitialisedApp {
                    window,
                    vk_instance,
                    texture_idx,
                });
            }
            StartCause::Poll => {}
            _ => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                debug_message!(debug_names::WINDOW_MANAGEMENT, "close requested");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                debug_message!(
                    debug_names::WINDOW_MANAGEMENT,
                    "window resized to {:?}x{:?}",
                    size.width,
                    size.height
                );

                match self {
                    App::Initialised(InitialisedApp { vk_instance, .. }) => {
                        vk_instance.get_command_buffers()
                    }
                    App::Uninitialised => {}
                }
            }
            _ => (),
        }
    }
}

fn main() {
    debug_message!(debug_names::GENERAL, "starting");

    let event_loop = EventLoop::new().unwrap();

    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);

    let mut app = App::default();

    let _ = event_loop.run_app(&mut app);
}
