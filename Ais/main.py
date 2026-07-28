import Ais.core.config as cfg
import argparse
import time
import os, sys

directory = os.path.join(os.path.dirname(__file__))
directory = directory[:directory.rfind("\\")]
sys.path.insert(0, os.path.abspath("../.."))
sys.path.append(directory)
cfg.root = os.path.join(os.path.dirname(__file__))

def run_ais():
    from Ais.core.window import Window
    from Ais.core.segmentation_editor import SegmentationEditor
    import Ais.core.settings as settings
    import glfw
    import imgui

    from imgui.integrations.glfw import GlfwRenderer
    import tkinter as tk

    tkroot = tk.Tk()
    tkroot.withdraw()

    if not glfw.init():
        raise Exception("Could not initialize GLFW library!")

    cfg.glfw_initialized = True
    # Init the main window, its imgui context, and a glfw rendering impl.
    main_window = Window(cfg.window_width, cfg.window_height, settings.ne_window_title)
    main_window.set_callbacks()
    main_window_imgui_context = imgui.create_context()
    main_window_imgui_glfw_implementation = GlfwRenderer(main_window.glfw_window)
    main_window.set_mouse_callbacks()
    main_window.set_window_callbacks()

    # set up editor
    segmentation_editor = SegmentationEditor(main_window, main_window_imgui_context, main_window_imgui_glfw_implementation)
    segmentation_editor.force_not_embedded()
    cfg.segmentation_editor = segmentation_editor
    main_window.set_icon(segmentation_editor.ICON)


    while not glfw.window_should_close(main_window.glfw_window):
        if not main_window.focused:
            glfw.poll_events()
        segmentation_editor.on_update()
        segmentation_editor.end_frame()


def main():
    parser = argparse.ArgumentParser(description="Ais headless CLI parser")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    segment_parser = subparsers.add_parser('segment', help='Segment .mrc volumes more efficiently in GUI-less mode.')
    segment_parser.add_argument('-m', '--model_path', required=True, type=str, help="Path to the model file")
    segment_parser.add_argument('-d', '--data', required=True, type=str, nargs='+', help="One or more directories, file paths, or glob patterns for .mrc files. Examples: '/data/volumes', 'volumes/035*.mrc volumes/036*.mrc', or explicit files 'volumes/035_001.mrc volumes/035_002.mrc'.")
    segment_parser.add_argument('-ou', '--output_directory', required=True, type=str, help="Directory to save the output")
    segment_parser.add_argument('-gpu', '--gpus', required=True, type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,3,4)")
    segment_parser.add_argument('-tta', '--test-time-augmentation', required=False, type=int, default=1, help="Integer between 1 and 8. If 1, no test time augmentation applied. If 2 - 8, differently oriented copies of the input tomogram are also segmented and the results averaged; orientations are [0, 90, 180, 270, 0*, 90*, 180*, 270*] (*=horizontal flip); e.g., when -tta 4, four samples of each tomogram are segmented, sampled with 0, 90, 180, and 270 deg. rotations relative to the original.")
    segment_parser.add_argument('-p', '--parallel', required=False, type=int, default=1, help="Integer 1 (default) or 0: whether to launch multiple parallel processes using one GPU each, or a single process using all GPUs.")
    segment_parser.add_argument('-overwrite', '--overwrite', required=False, type=int, default=0, help="If set to 1, tomograms for which a corresponding segmentation in the output_directory already exists are skipped (default 0).")
    segment_parser.add_argument('-apix', '--processing_apix', required=False, default=None, type=float, help="If set, override the model's trained scale (A/px) and process at specified value.")
    segment_parser.add_argument('-sigma', '--postprocessing-blur-sigma', required=False, type=float, nargs='+', default=(0.0, 0.0, 0.0), help="Gaussian filter sigma for postprocessing (in Angstrom). Either a single value (isotropic) or three separate values for the z, y, x dims. Default is no blur.")
    segment_parser.add_argument('--batch', required=False, type=int, default=1, help="Number of slices to batch together per inference call (default 1). Increase for faster inference if GPU memory allows.")
    segment_parser.add_argument('--workers', required=False, type=int, default=None, help="Number of CPU worker threads per GPU for preprocessing and postprocessing (default: cpu_count / n_gpus). Threads are shared dynamically between the two stages.")
    segment_parser.add_argument('--center', required=False, type=float, default=100.0, help="Percentage of the volume depth (Z) to segment, centered on the middle. E.g. --center 50 segments only the central 50%%. Default 100 (full volume).")

    pick_parser = subparsers.add_parser('pick', help='Pick particles using segmented volumes.')
    pick_parser.add_argument('-d', '--data_directory', required=True, type=str, help="Path to directory containing input segmentation .mrc's (e.g. /segmented/).")
    pick_parser.add_argument('-t', '--target', required=True, type=str, help="Feature to pick. For example, if segmented volumes are named '<tomogram_name>__Ribosome.mrc', '-t Ribosome' will select these.")
    pick_parser.add_argument('-ou', '--output_directory', required=False, type=str, default=None, help="Directory to save output coordinate files to. If left empty, will save to the input data directory.")
    pick_parser.add_argument('-m', '--margin', required=False, type=int, default=16, help="Margin (in pixels) to avoid picking particles close to tomogram edges.")
    pick_parser.add_argument('-threshold', required=False, type=float, default=128, help="Threshold to apply to volumes prior to finding local maxima (default 128).")
    pick_parser.add_argument('-b', '--binning', required=False, type=int, default=1, help="Binning factor to apply before processing (faster, possibly less accurate). Default is 1 (no binning)")
    pick_parser.add_argument('-spacing', required=False, type=float, default=10.0, help="Minimum distance between particles in Angstrom. Use ``-spacing-px`` to specify the minimum distance in voxel units instead.")
    pick_parser.add_argument('-spacing-px', required=False, type=float, default=None, help="Minimum distance between particles in px.")
    pick_parser.add_argument('-size', required=False, type=float, default=10.0, help="Minimum particle size in cubic Angstrom. Use ``-size-px`` to specify the minimum size in cubic voxel units instead.")
    pick_parser.add_argument('-size-px', required=False, type=float, default=None, help="Minimum particle size in number of voxels.")
    pick_parser.add_argument('-min-particles', required=False, type=int, default=0, help="Minimum number of particles that must be found in a tomogram for the output .star file to be saved. Default 0 (always save).")
    pick_parser.add_argument('-filament', required=False, action='store_true', help="If set, pick in filament mode rather than blob mode.")
    pick_parser.add_argument('--twist', required=False, type=float, default=0.0, help="If picking in filament mode, increment rlnAngleRot by this amount for every particle picked along a filament.")
    pick_parser.add_argument('-centroid', required=False, action='store_true', help="If set, if picking in blob mode, place coordinates at the centroid of each connected component rather than the deepest point. Only use when you are sure that particles are well separated!")
    pick_parser.add_argument('-orient', required=False, type=str, default=None, choices=['normal', 'long-axis'], help="In centroid mode, also write Euler angles derived from each blob's shape: 'normal' uses the disk normal (smallest principal axis), 'long-axis' uses the rod axis (largest principal axis). Sets rlnAngleTilt and rlnAnglePsi; rlnAngleRot is left at 0.")
    pick_parser.add_argument('-orient-sign', required=False, type=str, default='z', choices=['z', 'center', 'out'], help="Convention for resolving the sign ambiguity of the shape axis: 'z' forces a +z component, 'center'/'out' points the vector toward/away from the volume centre. Default: z.")
    pick_parser.add_argument('-length', required=False, type=float, default=500.0, help="Minimum filament length to place coordinates along (in Angstrom). Only used if -filament flag is set. Use ``-length-px`` to specify the length in pixels instead.")
    pick_parser.add_argument('-length-px', required=False, type=float, default=None, help="Minimum filament length to place coordinates along (in pixels). Only used if -filament flag is set.")
    pick_parser.add_argument('-p', '--parallel', required=False, type=int, default=1, help="Number of parallel picking processes to use (e.g. ``-p 64``, or however many threads your system can run at a time).")
    pick_parser.add_argument('-v', '--verbose', required=False, type=int, default=0, help="Verbose (1 or 0)")
    pick_parser.add_argument('--subset', required=False, type=str, default=None, help="Path to a .txt file listing tomogram names or paths (one per line), e.g. a Pom subset file. Only segmented volumes matching these tomograms will be picked.")

    train_parser = subparsers.add_parser('train', help='Train a model.')
    train_parser.add_argument('-a', '--model_architecture', required=False, type=str, default=None, help="Which model architecture to use, either as an integer index or as a model title (e.g. 13 or 'VGGNet M'). Use -models for a list of available architectures.")
    train_parser.add_argument('-m', '--model_path', required=False, type=str, default='', help="(Optional) path to a previously saved model to continue training. Overrides -a argument.")
    train_parser.add_argument('-t', '--training_data', required=False, type=str, nargs='+', help="Path(s) to the training data (.scnt) file(s). Multiple files may be given, e.g. '-t a.scnt b.scnt'; their samples are pooled during training. All files must share the same box size and depth.")
    train_parser.add_argument('-ou', '--output_directory', required=False, type=str, default='.', help="Directory to save the output")
    train_parser.add_argument('-gpu', '--gpus', required=False, default="0", type=str, help="Comma-separated list of GPU IDs to use (e.g., 0,1,4,5)")
    train_parser.add_argument('-e', '--epochs', required=False, type=int, default=50, help="Number of epochs to train the model for (default: 50).")
    train_parser.add_argument('-b', '--batch_size', required=False, type=int, default=32, help="Batch size to use during training (default: 32).")
    train_parser.add_argument('-c', '--copies', required=False, type=int, default=8, help="Number of augmented versions of the input images to include in the training data (all samples in different orientations). Default 8 (which would be the eight permutations of 90 degree rotations + horizontal flips; An argument >8 would include randomly rotated versions of the input images). If training data is 2.5D, augmentations 8 - 16 also include a flip in Z.")
    train_parser.add_argument('-r', '--rate', required=False, type=float, default=1e-3,help="Learning rate (default 1e-3)")
    train_parser.add_argument('-augment', required=False, action='store_true', help="If set, use extra scaling, contrast, brightness, and blurring augmentations.")
    train_parser.add_argument('-name', '--model_name', required=False, type=str, default="Unnamed model", help="Model name. File will be saved as output_directory/{name}.scnm")
    train_parser.add_argument('-models', '--model_architectures', required=False, action='store_true', help='List available model architectures.')

    extract_parsers = subparsers.add_parser('extract', help='Extract training data from annotated tomograms.')
    extract_parsers.add_argument('-d', '--data_directory', required=True, type=str, help="Directory containing annotated tomograms (.scns files).")
    extract_parsers.add_argument('-ou', '--output_directory', required=False, type=str, default='.', help="Directory to save the extracted training data (.scnt files).")
    extract_parsers.add_argument('-f', "--features", nargs="+", required=True, help="List of features to extract, e.g. 'Membrane Ribosome Microtubule'. A separate output file is created for each feature.")
    extract_parsers.add_argument('-size', "--box-size", required=False, type=int, default=128, help="Box size (in pixels) to extract. When not specified, box size is taken from the annotations.")
    extract_parsers.add_argument('-depth', "--box-depth", required=False, type=int, default=1, help="Box depth (in Z) to extract. Default 1 (2D). Must be odd - if not odd we add 1.")
    extract_parsers.add_argument('-bin', "--binning", required=False, type=int, default=1, help="Binning factor to apply (in XY). Output box size will be --box-size / --binning.")
    # margin is now computed automatically from the difference between --box-size and the annotation box size
    extract_parsers.add_argument('-e', "--exclude", required=False, type=str, nargs='+', default=None, help="Glob pattern or path to .txt file listing volumes to exclude from the extracted training data set..")
    extract_parsers.add_argument('-a', "--apix", default=None, required=False, type=float, help="Override the pixel size found in the tomogram header and use this value instead.")
    extract_parsers.add_argument('--merge', required=False, action='store_true', help="Combine all extracted training data into a single output file per feature, rather than one file per input volume.")
    extract_parsers.add_argument('--coordinates', required=False, action='store_true', help="Instead of exporting annotated training images, export just the box coordinates as a .star file.")
    extract_parsers.add_argument('--easymode', required=False, action='store_true', help=argparse.SUPPRESS)


    # feature_visualisation and audit are disabled / not shipped in this release.
    # fviz_parser = subparsers.add_parser('feature_visualisation', help="Synthesize input images that maximally activate a model's output feature (feature visualization).")
    # fviz_parser.add_argument('-m', '--model_path', required=True, type=str, help="Path to the model file (.scnm).")
    # fviz_parser.add_argument('-size', '--size', required=False, type=int, default=64, help="Side length (pixels) of the synthesized square images. Rounded up to a multiple of 32 (min 64). Default 64.")
    # fviz_parser.add_argument('-n', '--n_images', required=False, type=int, default=9, help="Number of times to synthesize the output neuron, each from a different random init (i.e. different facets of what the network reads as the feature). Default 9.")
    # fviz_parser.add_argument('-obj', '--objective', required=False, type=str, default='channel', choices=['channel', 'neuron'], help="What to maximize. 'channel' (default): each channel's mean activation over the whole image - uniform texture, no centre hotspot (the standard feature-viz objective). 'neuron': only the channel's centre pixel, so the feature lands at the centre (but the centre is prominent).")
    # fviz_parser.add_argument('-gpu', '--gpu', required=False, type=str, default="0", help="Comma-separated list of GPU IDs (e.g. 0,1,2,3). Neurons are split across the GPUs, one worker each. Default 0.")
    # fviz_parser.add_argument('-ou', '--output_directory', required=False, type=str, default='feature_visualisation', help="Directory to save the output .mrc stack and .png montage (default ./feature_visualisation).")
    # fviz_parser.add_argument('--highpass', required=False, type=float, default=24.0, help=argparse.SUPPRESS)
    # fviz_parser.add_argument('--iterations', required=False, type=int, default=400, help=argparse.SUPPRESS)
    # fviz_parser.add_argument('--seed', required=False, type=int, default=None, help=argparse.SUPPRESS)

    # if os.path.isdir('/cephfs/mlast'):
    #     audit_parser = subparsers.add_parser('audit', help='Audit training data against a trained model.')
    #     audit_parser.add_argument('-m', '--model', required=True, type=str, help="Path to model file (.scnm)")
    #     audit_parser.add_argument('-d', '--data', required=True, type=str, help="Path to training data file (.scnt)")
    #     audit_parser.add_argument('-gpu', '--gpu', required=False, type=int, default=0, help="GPU ID to use (default 0)")
    #     audit_parser.add_argument('-o', '--out', required=False, type=str, default='audit_output', help="Output directory (default: audit_output)")
    #     audit_parser.add_argument('--skip', action='store_true', help="Skip processing, reuse existing results, just launch the app")

    args, unknown = parser.parse_known_args()
    if args.command is None:
        run_ais()
    else:
        import Ais.core.cli_fn as aiscli

        if args.command == 'segment':
            gpus = [int(g) for g in args.gpus.split(',')]
            postprocessing_sigma = tuple(args.postprocessing_blur_sigma) if len(args.postprocessing_blur_sigma) == 3 else (args.postprocessing_blur_sigma[0],) * 3
            tta = args.test_time_augmentation
            aiscli.dispatch_parallel_segment(
                model_path=args.model_path,
                data_patterns=args.data,  # list of files (.mrc) and/or patterns (*) and/or dirs (remainder)
                output_directory=args.output_directory,
                gpus=gpus,
                test_time_augmentation=tta,
                parallel=args.parallel,
                overwrite=args.overwrite,
                processing_apix=args.processing_apix,
                postprocessing_sigma=postprocessing_sigma,
                batch_size=args.batch,
                n_workers=args.workers,
                center=args.center
            )

        elif args.command == 'pick':
            output_directory = args.output_directory if args.output_directory else args.data_directory
            aiscli.dispatch_parallel_pick(target=args.target,
                                          data_directory=args.data_directory,
                                          output_directory=output_directory,
                                          margin=args.margin,
                                          threshold=args.threshold,
                                          binning=args.binning,
                                          spacing=args.spacing,
                                          size=args.size,
                                          parallel=args.parallel,
                                          spacing_px=args.spacing_px,
                                          size_px=args.size_px,
                                          verbose=args.verbose==1,
                                          filament=args.filament,
                                          twist_per_sample=args.twist,
                                          filament_length=args.length,
                                          centroid=args.centroid,
                                          min_particles=args.min_particles,
                                          subset=args.subset,
                                          orient=args.orient,
                                          orient_sign=args.orient_sign)
        elif args.command == 'train':
            if args.model_architectures:
                aiscli.print_available_model_architectures()
                exit()
            else:
                aiscli.train_model(training_data=args.training_data,
                                   output_directory=args.output_directory,
                                   architecture=args.model_architecture,
                                   epochs=args.epochs,
                                   batch_size=args.batch_size,
                                   copies=args.copies,
                                   model_path=args.model_path,
                                   gpus=args.gpus,
                                   rate=args.rate,
                                   name=args.model_name,
                                   extra_augmentations=args.augment)
        elif args.command == 'extract':
            aiscli.extract_training_data(features=args.features,
                                         data_directory=args.data_directory,
                                         output_directory=args.output_directory,
                                         box_size=args.box_size,
                                         box_depth=args.box_depth,
                                         binning=args.binning,
                                         exclude=args.exclude,
                                         merge=args.merge,
                                         coordinates=args.coordinates,
                                         apix=args.apix,
                                         easymode=args.easymode)
        # feature_visualisation and audit are disabled / not shipped in this release.
        # elif args.command == 'feature_visualisation':
        #     from Ais.core.feature_vis import run_feature_visualisation
        #     run_feature_visualisation(model_path=args.model_path,
        #                               size=args.size,
        #                               n_images=args.n_images,
        #                               gpu=args.gpu,
        #                               output_directory=args.output_directory,
        #                               iterations=args.iterations,
        #                               seed=args.seed,
        #                               highpass=args.highpass,
        #                               objective=args.objective)
        # elif args.command == 'audit':
        #     from Ais.core.audit import run_audit
        #     run_audit(args.model, args.data, args.gpu, args.out, args.skip)

if __name__ == "__main__":
    main()


