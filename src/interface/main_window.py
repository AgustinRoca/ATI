import os
import time

import matplotlib.pyplot as plt
from PIL import ImageTk

from src.utils.mask_utils import *
from src.utils.noise_utils import *
from src.utils.filter_utils import *
from src.utils.threshold_utils import *
from src.utils.borders_utils import *
from src.utils.sift_utils import *
from src.interface.image_window import ImageWindow
from tkinter import *
from src.utils.image_utils import *
from numpy import copy
from tkinter import filedialog, messagebox
from src.utils.random_generator import (GaussianGenerator,
                                        RayleighGenerator,
                                        ExponentialGenerator)


class MainWindow:

    def __init__(self):
        self.root = Tk()
        self.root.title("ATI")

        # Menu 
        self.menubar = Menu(self.root)
        # Format of option list:
        #   (Label, function on click)
        #   None = separator
        image_menu_options = [('Load', self.open_file_window),
                              ('Save', self.save_image_window),
                              None,
                              ('Create Circle Image', self.create_circle),
                              ('Create Square Image', self.create_square),
                              ('Open Test Image', self.open_test),
                              None,
                              ('Exit', self.ask_quit)]
        edit_menu_options = [('Select', self.select),
                             None,
                             ('Get Pixel', self.get_pixel_value),
                             ('Modify Pixel', self.modify_pixel),
                             None,
                             ('Add', self.sum_images),
                             ('Subtract', self.subtract_images),
                             ('Multiply', self.multiply_images),
                             None,
                             ('Negative', self.image_negative),
                             ("Thresholding", self.image_thresholding),
                             ('Power function', self.power_gamma),
                             ('Equalize', self.equalize)]
        advanced_menu_options = [('Show histogram', self.show_hist),
                                 ('Show HSV histogram', self.show_hsv_hist)]
        noise_menu_options = [('Show Generator Histograms', self.show_gen_hist),
                              None,
                              ("Add Gaussian Noise", self.add_gaussian_noise),
                              ("Add Rayleigh Noise", self.add_rayleigh_noise),
                              ("Add Exponential Noise", self.add_exponential_noise),
                              ("Add Salt and Pepper Noise", self.add_salt_peppper_noise)]

        filter_menu_options = [('Mean filter', self.mean_filter),
                               ('Median filter', self.median_filter),
                               ('Gaussian mean filter', self.gaussian_filter),
                               ('Weighted median filter', self.weighted_median_filter),
                               ('Border filter', self.border_filter),
                               None,
                               ('Isotropic difussion', self.isotropic_difussion),
                               ('Anisotropic difussion', self.anisotropic_difussion),
                               ('Bilateral filter', self.bilateral)]

        border_detectors_menu_options = [('Prewitt', self.prewitt_detector),
                                         ('Sobel', self.sobel_detector),
                                         ('Directional', self.directional_detector),
                                         ('Laplacian', self.laplacian_detector),
                                         ('Laplacian with threshold', self.laplacian_detector_with_threshold),
                                         ('LoG', self.laplacian_o_gauss_detector),
                                         None,
                                         ('Prewitt (vertical)', self.prewitt_vertical_filter),
                                         ('Prewitt (horizontal)', self.prewitt_horizontal_filter),
                                         None,
                                         ('Sobel (vertical)', self.sobel_vertical_filter),
                                         ('Sobel (horizontal)', self.sobel_horizontal_filter),
                                         None,
                                         ('Canny', self.canny_border),
                                         ('S.U.S.A.N.', self.susan_border),
                                         ('Hough Transformation', self.hough_transformation),
                                         ('Active Contours', self.active_contours),
                                         ('Active Contours video', self.active_contours_video),
                                         None,
                                         ('Harris', self.harris_corners)]

        threshold_menu_option = [('Global', self.global_thresholding),
                                 ("Otsu", self.otsu_thresholding)]

        object_detectors_options = [('S.I.F.T Key Points', self.key_points),
                                    ('S.I.F.T', self.sift),
                                    ('S.I.F.T Video', self.sift_video)]

        menu_options = {'Image': image_menu_options,
                        'Edit': edit_menu_options,
                        'Advanced': advanced_menu_options,
                        "Noise": noise_menu_options,
                        "Filter": filter_menu_options,
                        "Border Detector": border_detectors_menu_options,
                        'Object Detector': object_detectors_options,
                        'Threshold': threshold_menu_option}

        for option in menu_options.keys():
            self._add_to_menu(option, menu_options[option])

        self.frame = Frame(
            self.root,
            height=self.root.winfo_screenheight() / 8,
            width=self.root.winfo_screenwidth(),
        )
        self.frame.pack()
        Label(self.frame, text="Welcome to ATI Program").pack()
        self.root.config(menu=self.menubar)

        self.root.pack_propagate(0)

        self.windows = []
        self.unsaved_imgs = {}

        self.root.mainloop()

    # Open file window
    def open_file_window(self):
        filename = filedialog.askopenfilename(initialdir="./photos", title="Select an Image",
                                              filetypes=(("raw", "*.RAW"),
                                                         ("pgm", "*.pgm"),
                                                         ("ppm", "*.ppm"),
                                                         ("jpg", "*.jpg"),
                                                         ("png", "*.png"),
                                                         ("jpeg", "*.jpeg")))
        if not filename:
            return

        w, h = None, None
        if filename.lower().endswith('.raw'):
            w, h = self.ask_width_and_height()
        ImageWindow(self, load(filename, w, h), filename)

    # Save Image
    def save_image_window(self):
        if len(self.unsaved_imgs) == 0:
            top = Toplevel()
            Label(top, text="No photo to save").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            (title, img) = self.select_save_window()
            filename = filedialog.asksaveasfilename(initialdir="./photos", title="Save As")
            if not filename:
                return
            save(img, filename)

            self.unsaved_imgs.pop(title)

            top = Toplevel()
            Label(top, text="Photo saved successfully").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    # Create Image with white Circle in the middle
    def create_circle(self):
        window = ImageWindow(self, np.asarray(bin_circle()))
        self.unsaved_imgs[window.title] = window.img

    # Create Image with white Square in the middle
    def create_square(self):
        window = ImageWindow(self, np.asarray(bin_rectangle()))
        self.unsaved_imgs[window.title] = window.img

    def open_test(self):
        window = ImageWindow(self, load('./photos/test_images/pgm/test.pgm'), './photos/test_images/pgm/test.pgm')
        self.unsaved_imgs[window.title] = window.img

    # Get pixel value
    def get_pixel_value(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            x, y = self.ask_xy('Enter (x,y) coordinates of pixel')
            value = get_pixel(img, x, y)

            window = Toplevel()
            frame = Frame(window)
            frame.pack()
            Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(value)) \
                .grid(row=0, column=0, columnspan=3)

            Button(frame, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)

    # Modify pixel TODO: Support RGB values in colored images
    def modify_pixel(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = copy(self.select_img_from_windows())
            x, y, new_value = self.ask_xy_new('Enter (x,y) coordinates of pixel')
            put_pixel(img, x, y, new_value)
            window = ImageWindow(self, img)
            info_window = Toplevel()
            frame = Frame(info_window)
            frame.pack()
            Label(frame, text="Pixel's value on (" + str(x) + ", " + str(y) + "): " + str(new_value)) \
                .grid(row=0, column=0, columnspan=3)
            Button(frame, text="Done", command=info_window.destroy, padx=20).grid(row=2, column=1)
            self.unsaved_imgs[window.title] = window.img

    # Add two images
    def sum_images(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            windows = self.select_from_to_windows()
            new_img = add(windows[0].img, windows[1].img)
            img_window = ImageWindow(self, new_img)
            self.unsaved_imgs[img_window.title] = img_window.img

    # Subtract two images
    def subtract_images(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            windows = self.select_from_to_windows()
            new_img = subtract(windows[0].img, windows[1].img)
            img_window = ImageWindow(self, new_img)
            self.unsaved_imgs[img_window.title] = img_window.img

    # Multiply two images
    def multiply_images(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            windows = self.select_from_to_windows()
            new_img = multiply(windows[0].img, windows[1].img)
            img_window = ImageWindow(self, new_img)
            self.unsaved_imgs[img_window.title] = img_window.img

    # Get image's negaive        
    def image_negative(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = negative(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def image_thresholding(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            new_img = thresholding(threshold, img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def power_gamma(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            gamma = self.ask_for_gamma()
            if (0 < gamma < 2) and gamma != 1:
                new_img = power(img, gamma)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                    title="Error", message="Gamma should be between 0 and 2, and different from 1."
                )

    def equalize(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = equalize(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    # Exit
    def ask_quit(self):
        if messagebox.askokcancel("Quit", "Are you sure you want to quit?"):
            self.root.destroy()

    # Show RGB and gray histograms
    def show_hist(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            plot_hist_rgb(img)

    # Show HSV histograms
    def show_hsv_hist(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            converted = rgb_to_hsv(img)
            plot_hist_hsv(converted)

    # Shows histogram for a random number generator
    def show_gen_hist(self):
        dist = self.ask_distribution()
        if dist == "Gaussian":
            mean, std = self.ask_gaussian_args()
            plot_gen_hist(GaussianGenerator(mean, std, 1000), dist)
        elif dist == "Rayleigh":
            phi = self.ask_rayleigh_args()
            plot_gen_hist(RayleighGenerator(phi, 1000), dist)
        elif dist == "Exponential":
            lam = self.ask_exponential_args()
            plot_gen_hist(ExponentialGenerator(lam, 1000), dist)

    # Add Gaussian Noise to image
    def add_gaussian_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                mean, std = self.ask_gaussian_args()

                size = img.size

                new_img = apply_noise(img, GaussianGenerator(mean, std, size), threshold, True)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                    title="Error", message="Threshold should be between 0 and 1."
                )

    # Add Rayleigh Noise to image
    def add_rayleigh_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                phi = self.ask_rayleigh_args()

                size = img.size

                new_img = apply_noise(img, RayleighGenerator(phi, size), threshold, False)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                    title="Error", message="Threshold should be between 0 and 1."
                )

    # Add Exponential Noise to image
    def add_exponential_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                lam = self.ask_exponential_args()

                size = img.size

                new_img = apply_noise(img, ExponentialGenerator(lam, size), threshold, False)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                    title="Error", message="Threshold should be between 0 and 1."
                )

    # Add Salt and Pepper Noise to image
    def add_salt_peppper_noise(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            if 0.0 <= threshold <= 1.0:
                new_img = apply_salt_pepper_noise(img, threshold)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                    title="Error", message="Threshold should be between 0 and 1."
                )

    def mean_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            mask_size = self.ask_for_mask_size()
            new_img = mean_filter(img, mask_size)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def median_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            mask_size = self.ask_for_mask_size()
            new_img = median_filter(img, mask_size)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def gaussian_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            deviation = self.ask_for_deviation()
            new_img = gaussian_filter(img, deviation)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def weighted_median_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = weighted_median_filter(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def border_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            mask_size = self.ask_for_mask_size()
            new_img = border_filter(img, mask_size)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def prewitt_vertical_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = prewitt_vertical_mask(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def prewitt_horizontal_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = prewitt_horizontal_mask(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def prewitt_detector(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = prewitt_detector(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def sobel_vertical_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = sobel_vertical_mask(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def sobel_horizontal_filter(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = sobel_horizontal_mask(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def sobel_detector(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = sobel_detector(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def directional_detector(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            dir = self.select_direction()
            new_img = directional_detector(img, dir)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def laplacian_detector(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            new_img = laplacian_detector(img)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def laplacian_detector_with_threshold(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            new_img = laplacian_detector(img, threshold)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def laplacian_o_gauss_detector(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            deviation, threshold = self.ask_for_deviation_and_threshold()
            new_img = laplacian_o_gauss_detector(img, deviation, threshold)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def isotropic_difussion(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            t = self.ask_isotropic_args()
            new_img = isotropic_dif(img, t)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def anisotropic_difussion(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            t, sigma = self.ask_anisotropic_args()
            new_img = anisotropic_diff(img, sigma, t)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def bilateral(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            s, r = self.ask_sigmas_args()
            size = self.ask_for_mask_size()
            new_img = bilateral_filter(img, r, s, size)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

    def global_thresholding(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            t = 0
            while (t <= 0 or t >= 255):
                t = self.ask_for_threshold()
            if len(img.shape) > 2:
                real_t = np.zeros(len(img[0][0]))
                for k in range(0, len(img[0][0])):
                    real_t[k] = global_threshold(img[:, :, k], t)
                new_img = thresholding_color(real_t, img)
            else:
                real_t = global_threshold(img, t)
                new_img = thresholding(real_t, img)

            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img
            top = Toplevel()
            Label(top, text="Threshold: " + str(real_t)).grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    def otsu_thresholding(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            if len(img.shape) > 2:
                real_t = np.zeros(len(img[0][0]))
                for k in range(0, len(img[0][0])):
                    real_t[k] = otsu_threshold(img[:, :, k])
                new_img = thresholding_color(real_t, img)
            else:
                real_t = otsu_threshold(img)
                new_img = thresholding(real_t, img)

            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img
            top = Toplevel()
            Label(top, text="Threshold: " + str(real_t)).grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)

    def susan_border(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            t, together = self.ask_susan_args()
            if together:
                new_img = susan_filter_together(img, t)
                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                border_img, corner_img = susan_filter_apart(img, t)
                window1 = ImageWindow(self, border_img)
                window2 = ImageWindow(self, corner_img)
                self.unsaved_imgs[window1.title] = window1.img
                self.unsaved_imgs[window2.title] = window2.img


    def canny_border(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            t1, t2 = self.ask_canny_args()
            new_img = canny(img, t1, t2)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img


    def active_contours(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img_window = self.select_window()
            img = img_window.img
            copy_window = ImageWindow(self, copy(img))
            left, up, right, down = self.get_selection(copy_window)
            epsilon, max_iterations = self.ask_active_contours_args()
            _, mean = pixels_info(img[up:down + 1, left:right + 1])
            new_img = active_contours(img, left, up, right, down, mean, epsilon, max_iterations)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img


    def active_contours_video(self):
        imgs = self.select_video()
        showing_img_window = ImageWindow(self, imgs[0])
        first_img = showing_img_window.img
        left, up, right, down = self.get_selection(showing_img_window)
        epsilon, max_iterations = self.ask_active_contours_args()
        _, mean = pixels_info(first_img[up:down + 1, left:right + 1])
        prev_curve, lin, lout = initialize_marks(first_img, left, up, right, down)
        showing_img_window = ImageWindow(self, imgs[0])
        self.root.after(10, self.update_active_contours_video, imgs, 0, prev_curve, lin, lout, mean, epsilon,
                        max_iterations,
                        showing_img_window)


    def update_active_contours_video(self, imgs, current_i, prev_curve, lin, lout, mean, epsilon, max_iterations,
                                     window):
        img = imgs[current_i]
        new_img, prev_curve, lin, lout = active_contours_vid(img, prev_curve, lin, lout, mean, epsilon, max_iterations)
        window.change_image(new_img)
        if current_i < len(imgs) - 1:
            self.root.after(10, self.update_active_contours_video, imgs, current_i + 1, prev_curve, lin, lout, mean,
                            epsilon, max_iterations, window)


    def sift_video(self):
        filenames, img_filename = self.select_video_filenames()
        print(img_filename)
        n, threshold, show = self.ask_for_sift_arg()
        showing_img_window = ImageWindow(self, load(filenames[0]))
        matches_list = []
        self.root.after(100, self.update_sift_video, img_filename, filenames, 0, threshold, n, show, showing_img_window, matches_list)


    def update_sift_video(self, img_filename, filenames, current_i, threshold, n, show, window, matches_list, kp1=None, d1=None):
        new_img, matches, kp1, d1 = sift_detector(img_filename, filenames[current_i], threshold, n, show, kp1, d1)
        matches_list.append(matches)
        window.change_image(new_img)
        if current_i < len(filenames) - 1:
            self.root.after(100, self.update_sift_video, img_filename, filenames, current_i + 1, threshold, n, show, window, matches_list, kp1, d1)
        else:
            plt.plot(matches_list)
            plt.show()


    def hough_transformation(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            t1, t2 = self.ask_canny_args()
            epsilon, theta, rho = self.ask_hough_args()

            new_img = hough_linear(img, epsilon, t1, t2, theta, rho)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

            
    def harris_corners(self):
        if len(self.windows) == 0:
            window = Toplevel()
            Label(window, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            img = self.select_img_from_windows()
            threshold = self.ask_for_threshold()
            new_img = harris(img, threshold)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img

        

    def key_points(self):
            filename = self.select_filename_from_windows()
            new_img = key_points(filename)
            window = ImageWindow(self, new_img)
            self.unsaved_imgs[window.title] = window.img


    def sift(self):
        if len(self.windows) < 2:
            window = Toplevel()
            Label(window, text="Not enough images, please load at least 2").grid(row=0, column=0, columnspan=3)
            Button(window, text="Done", command=window.destroy, padx=20).grid(row=2, column=1)
        else:
            filename_1, filename_2 = self.select_filenames_from_windows()
            n, threshold, show = self.ask_for_sift_arg()
            if 0.0 <= n <= 1.0:
                new_img, matches, _, _ = sift(filename_1, filename_2, threshold, show)
                info_window = Toplevel()
                frame = Frame(info_window)
                frame.pack()
                if (matches >= n):
                    Label(frame, text="They are the same image with " + str(matches * 100) + "% matches.")\
                        .grid(row=0, column=0, columnspan=3)
                    Button(frame, text="Done", command=info_window.destroy, padx=20).grid(row=2, column=1)
                else:
                    Label(frame, text="They are not the same image. (Matches: " + str(matches * 100) + "%)")\
                        .grid(row=0, column=0, columnspan=3)
                    Button(frame, text="Done", command=info_window.destroy, padx=20).grid(row=2, column=1)

                window = ImageWindow(self, new_img)
                self.unsaved_imgs[window.title] = window.img
            else:
                messagebox.showerror(
                    title="Error", message="Matches percentege should be between 0 and 1."
                )
            

    # Selection mode
    def select(self):
        if len(self.windows) == 0:
            top = Toplevel()
            Label(top, text="No Image, please load one").grid(row=0, column=0, columnspan=3)
            Button(top, text="Done", command=top.destroy, padx=20).grid(row=2, column=1)
        else:
            image_window = self.select_window()
            image_window.selection_mode()

    # AUXILIARY FUNCTIONS
    def _add_to_menu(self, label, image_menu_options):
        menu = Menu(self.menubar, tearoff=0)
        for option in image_menu_options:
            if option is None:
                menu.add_separator()
            else:
                menu.add_command(label=option[0], command=option[1])
        self.menubar.add_cascade(label=label, menu=menu)

    def select_img_from_windows(self):
        window = self.select_window()
        return window.img

    def select_filenames_from_windows(self):
            window_1, window_2 = self.select_2_windows()
            return window_1.filename, window_2.filename


    def select_filename_from_windows(self):
            window = self.select_window()
            return window.filename

        
    def select_window(self):
        if len(self.windows) == 1:
            return self.windows[0]
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select image").grid(row=0, column=0, columnspan=3)

        clicked = StringVar()
        options = self.get_windows_titles()

        clicked.set(options[0])

        op_menu = OptionMenu(frame, clicked, *options)
        op_menu.grid(row=1, column=0, columnspan=2)

        window_name_var = StringVar()
        Button(frame, text="Select", command=(lambda: window_name_var.set(clicked.get()))).grid(row=1, column=2)
        frame.wait_variable(window_name_var)
        window_name = window_name_var.get()
        window.destroy()
        for image_window in self.windows:
            if image_window.title == window_name:
                return image_window
    

    def select_2_windows(self):
        if len(self.windows) == 2:
            return self.windows[0], self.windows[1]
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select 2 images").grid(row=0, column=0, columnspan=3)

        clicked_1 = StringVar()
        clicked_2 = StringVar()

        options_1 = self.get_windows_titles()
        options_2 = self.get_windows_titles()

        clicked_1.set(options_1[0])
        clicked_2.set(options_1[0])

        op_menu_1 = OptionMenu(frame, clicked_1, *options_1)
        op_menu_1.grid(row=1, column=0, columnspan=2)
        op_menu_2 = OptionMenu(frame, clicked_2, *options_2)
        op_menu_2.grid(row=2, column=0, columnspan=2)


        window_name_var_1 = StringVar()
        window_name_var_2 = StringVar()

        Button(frame, text="Select", command= (lambda: (window_name_var_1.set(clicked_1.get()),
                                    window_name_var_2.set(clicked_2.get())))).grid(row=2, column=2)
        frame.wait_variable(window_name_var_1)
        window_name_1 = window_name_var_1.get()
        window_name_2 = window_name_var_2.get()
        window.destroy()
        image_window_1 = None
        image_window_2 = None
        for image_window in self.windows:
            if image_window.title == window_name_1:
                image_window_1 = image_window
            if image_window.title == window_name_2:
                image_window_2 = image_window

        return image_window_1, image_window_2

    def select_save_window(self):
        if len(self.unsaved_imgs) == 1:
            return list(self.unsaved_imgs.items())[0]
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select image").grid(row=0, column=0, columnspan=3)

        clicked = StringVar()
        options = list(self.unsaved_imgs.keys())

        clicked.set(options[0])

        op_menu = OptionMenu(frame, clicked, *options)
        op_menu.grid(row=1, column=0, columnspan=2)

        window_name_var = StringVar()
        Button(frame, text="Select", command=(lambda: window_name_var.set(clicked.get()))).grid(row=1, column=2)
        frame.wait_variable(window_name_var)
        window_name = window_name_var.get()
        window.destroy()
        for image_window in self.unsaved_imgs.keys():
            if image_window == window_name:
                return (image_window, self.unsaved_imgs.get(image_window))

    def select_from_to_windows(self):
        if len(self.windows) == 1:
            return (self.windows[0], self.windows[0])
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select image 1").grid(row=0, column=0, columnspan=3)

        clicked_from = StringVar()
        options = self.get_windows_titles()

        clicked_from.set(options[0])

        op_menu_from = OptionMenu(frame, clicked_from, *options)
        op_menu_from.grid(row=1, column=0, columnspan=2)

        Label(frame, text="Select image 2").grid(row=2, column=0, columnspan=3)

        clicked_to = StringVar()
        options = self.get_windows_titles()

        clicked_to.set(options[0])

        op_menu_to = OptionMenu(frame, clicked_to, *options)
        op_menu_to.grid(row=3, column=0, columnspan=2)

        window_name_var_from = StringVar()
        window_name_var_to = StringVar()

        Button(frame, text="Select", command=(lambda: self.set_vars(window_name_var_from,
                                                                    clicked_from.get(),
                                                                    window_name_var_to,
                                                                    clicked_to.get()))).grid(row=3, column=2)

        frame.wait_variable(window_name_var_to)
        window_from_name = window_name_var_from.get()
        window_to_name = window_name_var_to.get()
        window.destroy()

        window_from = None
        window_to = None
        for image_window in self.windows:
            if image_window.title == window_from_name:
                window_from = image_window
            if image_window.title == window_to_name:
                window_to = image_window
        return (window_from, window_to)

    def set_vars(self, from_var, from_val, to_var, to_val):
        from_var.set(from_val)
        to_var.set(to_val)

    def get_windows_titles(self):
        titles = []
        for window in self.windows:
            titles.append(window.title)
        return titles

    # User input windows
    @staticmethod
    def ask_xy(window_title):
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text=window_title).grid(row=0, column=0, columnspan=3)

        Label(frame, text="Enter x:").grid(row=1, column=0)
        x_entry = Entry(frame, width=10)
        x_entry.grid(row=2, column=0)

        Label(frame, text="Enter y:").grid(row=1, column=1)
        y_entry = Entry(frame, width=10)
        y_entry.grid(row=2, column=1)

        x_var = IntVar()
        y_var = IntVar()

        button = Button(frame, text="Enter",
                        command=(lambda: (x_var.set(int(x_entry.get())), y_var.set(int(y_entry.get())))),
                        padx=20)
        button.grid(row=2, column=2)

        frame.wait_variable(y_var)
        x, y = x_var.get(), y_var.get()
        window.destroy()
        return x, y

    @staticmethod
    def ask_xy_new(window_title):
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text=window_title).grid(row=0, column=0, columnspan=3)

        Label(frame, text="Enter x:").grid(row=1, column=0)
        x_entry = Entry(frame, width=10)
        x_entry.grid(row=2, column=0)

        Label(frame, text="Enter y:").grid(row=1, column=1)
        y_entry = Entry(frame, width=10)
        y_entry.grid(row=2, column=1)

        Label(frame, text="Enter new value:").grid(row=3, column=0, columnspan=2)
        new_entry = Entry(frame, width=20)
        new_entry.grid(row=4, column=0, columnspan=2)

        x_var = IntVar()
        y_var = IntVar()
        new_var = IntVar()

        button = Button(frame, text="Enter",
                        command=(lambda: (x_var.set(int(x_entry.get())),
                                          y_var.set(int(y_entry.get())),
                                          new_var.set(int(new_entry.get())))),
                        padx=20)
        button.grid(row=4, column=2)

        frame.wait_variable(new_var)
        x, y, new = x_var.get(), y_var.get(), new_var.get()
        window.destroy()
        return x, y, new

    @staticmethod
    def ask_width_and_height():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter image width:").grid(row=0, column=0)
        width_entry = Entry(frame, width=10)
        width_entry.grid(row=1, column=0)

        Label(frame, text="Enter image height:").grid(row=0, column=1)
        height_entry = Entry(frame, width=10)
        height_entry.grid(row=1, column=1)

        width_var = IntVar()
        height_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (height_var.set(int(height_entry.get())), width_var.set(int(width_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(width_var)
        w, h = width_var.get(), height_var.get()
        window.destroy()
        return w, h

    @staticmethod
    def ask_for_threshold():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter threshold:").grid(row=0, column=0)
        threshold_entry = Entry(frame, width=10)
        threshold_entry.grid(row=1, column=0)

        threshold_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (threshold_var.set(float(threshold_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(threshold_var)
        threshold = threshold_var.get()
        window.destroy()
        return threshold

    @staticmethod
    def ask_for_gamma():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter gamma:").grid(row=0, column=0)
        gamma_entry = Entry(frame, width=10)
        gamma_entry.grid(row=1, column=0)

        gamma_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (gamma_var.set(float(gamma_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(gamma_var)
        gamma = gamma_var.get()
        window.destroy()
        return gamma

    @staticmethod
    def ask_gaussian_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter mean:").grid(row=0, column=0)
        mean_entry = Entry(frame, width=10)
        mean_entry.grid(row=1, column=0)

        Label(frame, text="Enter standard deviation:").grid(row=0, column=1)
        std_entry = Entry(frame, width=10)
        std_entry.grid(row=1, column=1)

        mean_var = DoubleVar()
        std_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (mean_var.set(float(mean_entry.get())), std_var.set(float(std_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(std_var)
        m, s = mean_var.get(), std_var.get()
        window.destroy()
        return m, s

    @staticmethod
    def ask_rayleigh_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter phi:").grid(row=0, column=0)
        phi_entry = Entry(frame, width=10)
        phi_entry.grid(row=1, column=0)

        phi_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (phi_var.set(float(phi_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(phi_var)
        phi = phi_var.get()
        window.destroy()
        return phi

    @staticmethod
    def ask_exponential_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter lambda:").grid(row=0, column=0)
        lambda_entry = Entry(frame, width=10)
        lambda_entry.grid(row=1, column=0)

        lambda_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (lambda_var.set(float(lambda_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(lambda_var)
        lam = lambda_var.get()
        window.destroy()
        return lam

    @staticmethod
    def ask_distribution():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select Distribution").grid(row=0, column=0, columnspan=3)

        clicked = StringVar()
        options = ["Gaussian", "Rayleigh", "Exponential"]

        clicked.set(options[0])

        op_menu = OptionMenu(frame, clicked, *options)
        op_menu.grid(row=1, column=0, columnspan=2)

        dist_var = StringVar()
        Button(frame, text="Select", command=(lambda: dist_var.set(clicked.get()))).grid(row=1, column=2)
        frame.wait_variable(dist_var)
        dist_val = dist_var.get()
        window.destroy()
        return dist_val

    @staticmethod
    def ask_for_mask_size():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter mask side:").grid(row=0, column=0)
        side_entry = Entry(frame, width=10)
        side_entry.grid(row=1, column=0)

        side_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (side_var.set(int(side_entry.get())))),
                        padx=20)
        button.grid(row=1, column=1)

        frame.wait_variable(side_var)
        side = side_var.get()
        window.destroy()
        return side

    @staticmethod
    def ask_for_deviation():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter standard deviation:").grid(row=0, column=1)
        std_entry = Entry(frame, width=10)
        std_entry.grid(row=1, column=1)

        std_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(lambda: (std_var.set(float(std_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(std_var)
        std = std_var.get()
        window.destroy()
        return std

    @staticmethod
    def ask_for_sift_arg():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter threshold for distance:").grid(row=0, column=1)
        threshold_entry = Entry(frame, width=10)
        threshold_entry.grid(row=1, column=1)

        Label(frame, text="Enter threshold of matches:").grid(row=2, column=1)
        n_entry = Entry(frame, width=10)
        n_entry.grid(row=3, column=1)

        Label(frame, text="Enter number of matches to show:").grid(row=4, column=1)
        show_entry = Entry(frame, width=10)
        show_entry.grid(row=5, column=1)

        n_var = DoubleVar()
        threshold_var = DoubleVar()
        show_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (n_var.set(float(n_entry.get())),
                                    threshold_var.set(float(threshold_entry.get())),
                                     show_var.set(int(show_entry.get())))),
                        padx=20)
        button.grid(row=5, column=2)
        
        frame.wait_variable(n_var)
        n = n_var.get()
        threshold = threshold_var.get()
        show = show_var.get()
        window.destroy()
        return n, threshold, show


    @staticmethod
    def ask_for_sift_video_arg():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter threshold for distance:").grid(row=0, column=1)
        threshold_entry = Entry(frame, width=10)
        threshold_entry.grid(row=1, column=1)


        Label(frame, text="Enter number of matches to show:").grid(row=2, column=1)
        show_entry = Entry(frame, width=10)
        show_entry.grid(row=3, column=1)

        threshold_var = DoubleVar()
        show_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (threshold_var.set(float(threshold_entry.get())),
                                     show_var.set(int(show_entry.get())))),
                        padx=20)
        button.grid(row=3, column=2)
        
        frame.wait_variable(threshold_var)
        threshold = threshold_var.get()
        show = show_var.get()
        window.destroy()
        return threshold, show


    @staticmethod
    def ask_isotropic_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter time:").grid(row=0, column=1)
        t_entry = Entry(frame, width=10)
        t_entry.grid(row=1, column=1)

        t_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(lambda: (t_var.set(float(t_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(t_var)
        t = t_var.get()
        window.destroy()

        return t

    @staticmethod
    def ask_anisotropic_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter time:").grid(row=0, column=0)
        t_entry = Entry(frame, width=10)
        t_entry.grid(row=1, column=0)

        Label(frame, text="Enter sigma:").grid(row=0, column=1)
        sigma_entry = Entry(frame, width=10)
        sigma_entry.grid(row=1, column=1)

        t_var = IntVar()
        sigma_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (t_var.set(float(t_entry.get())),
                                     sigma_var.set(int(sigma_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(sigma_var)
        t, sigma = t_var.get(), sigma_var.get()
        window.destroy()
        return t, sigma

    @staticmethod
    def ask_sigmas_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter sigma s:").grid(row=0, column=0)
        s_entry = Entry(frame, width=10)
        s_entry.grid(row=1, column=0)

        Label(frame, text="Enter sigma r:").grid(row=0, column=1)
        r_entry = Entry(frame, width=10)
        r_entry.grid(row=1, column=1)

        s_var = DoubleVar()
        r_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (s_var.set(float(s_entry.get())),
                                     r_var.set(int(r_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(r_var)
        s, r = s_var.get(), r_var.get()
        window.destroy()
        return s, r

    @staticmethod
    def select_direction():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Select Direction").grid(row=0, column=0, columnspan=3)

        clicked = StringVar()
        options = ["Horizontal", "45º", "Vertical", "135º"]

        clicked.set(options[0])

        op_menu = OptionMenu(frame, clicked, *options)
        op_menu.grid(row=1, column=0, columnspan=2)

        dir_var = StringVar()
        Button(frame, text="Select", command=(lambda: dir_var.set(clicked.get()))).grid(row=1, column=2)
        frame.wait_variable(dir_var)
        dir_val = dir_var.get()
        if dir_val == "Horizontal":
            dir_val = 0
        elif dir_val == "45º":
            dir_val = 45
        elif dir_val == "Vertical":
            dir_val = 90
        elif dir_val == "135º":
            dir_val = 135
        window.destroy()
        return dir_val

    @staticmethod
    def ask_for_deviation_and_threshold():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter deviation:").grid(row=0, column=0)
        deviation_entry = Entry(frame, width=10)
        deviation_entry.grid(row=1, column=0)

        Label(frame, text="Enter threshold:").grid(row=0, column=1)
        threshold_entry = Entry(frame, width=10)
        threshold_entry.grid(row=1, column=1)

        deviation_var = DoubleVar()
        threshold_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (deviation_var.set(float(deviation_entry.get())),
                                     threshold_var.set(int(threshold_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(threshold_var)
        deviation, threshold = deviation_var.get(), threshold_var.get()
        window.destroy()
        return deviation, threshold

    @staticmethod
    def ask_susan_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()

        Label(frame, text="Enter t:").grid(row=0, column=1)
        t_entry = Entry(frame, width=10)
        t_entry.grid(row=1, column=1)

        t_var = IntVar()
        together_var = BooleanVar()

        Label(frame, text="Borders and corners on").grid(row=0, column=2)
        button_together = Button(frame,
                                 text="Same Image",
                                 command=(
                                     lambda: (t_var.set(float(t_entry.get())),
                                              together_var.set(True))),
                                 padx=20)
        button_together.grid(row=1, column=2)

        button_separate = Button(frame,
                                 text="Separate Images",
                                 command=(
                                     lambda: (t_var.set(float(t_entry.get())),
                                              together_var.set(False))),
                                 padx=20)
        button_separate.grid(row=2, column=2)

        frame.wait_variable(t_var)
        t = t_var.get()
        together = together_var.get()
        window.destroy()

        return t, together

    @staticmethod
    def ask_canny_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter t1:").grid(row=0, column=0)
        t1_entry = Entry(frame, width=10)
        t1_entry.grid(row=1, column=0)

        Label(frame, text="Enter t2:").grid(row=0, column=1)
        t2_entry = Entry(frame, width=10)
        t2_entry.grid(row=1, column=1)

        t1_var = DoubleVar()
        t2_var = DoubleVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (t1_var.set(float(t1_entry.get())),
                                     t2_var.set(float(t2_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(t2_var)
        t1, t2 = t1_var.get(), t2_var.get()
        window.destroy()
        return t1, t2

    @staticmethod
    def ask_hough_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter epsilon:").grid(row=0, column=0)
        epsilon_entry = Entry(frame, width=10)
        epsilon_entry.grid(row=1, column=0)

        Label(frame, text="Enter theta step:").grid(row=0, column=1)
        theta_entry = Entry(frame, width=10)
        theta_entry.grid(row=1, column=1)

        Label(frame, text="Enter rho step:").grid(row=0, column=2)
        rho_entry = Entry(frame, width=10)
        rho_entry.grid(row=1, column=2)

        epsilon_var = DoubleVar()
        theta_var = DoubleVar()
        rho_var = DoubleVar()

        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (epsilon_var.set(float(epsilon_entry.get())),
                                     theta_var.set(float(theta_entry.get())),
                                     rho_var.set(float(rho_entry.get())))),

                        padx=20)
        button.grid(row=1, column=3)

        frame.wait_variable(theta_var)
        epsilon, theta, rho = epsilon_var.get(), theta_var.get(), rho_var.get()
        window.destroy()
        return epsilon, theta, rho

    @staticmethod
    def ask_active_contours_args():
        window = Toplevel()
        frame = Frame(window)
        frame.pack()
        Label(frame, text="Enter epsilon:").grid(row=0, column=0)
        e_entry = Entry(frame, width=10)
        e_entry.grid(row=1, column=0)

        Label(frame, text="Enter max iterations:").grid(row=0, column=1)
        max_entry = Entry(frame, width=10)
        max_entry.grid(row=1, column=1)

        e_var = IntVar()
        max_var = IntVar()
        button = Button(frame,
                        text="Enter",
                        command=(
                            lambda: (e_var.set(int(e_entry.get())),
                                     max_var.set(int(max_entry.get())))),
                        padx=20)
        button.grid(row=1, column=2)

        frame.wait_variable(max_var)
        epsilon, max_iterations = e_var.get(), max_var.get()
        window.destroy()
        return epsilon, max_iterations

    @staticmethod
    def get_selection(img):
        return img.selection_mode(select_only=True)

    def select_video(self):
        filenames = self.select_folder()
        imgs = []
        for filename in filenames:
            imgs.append(load(filename))
        return imgs

    def select_video_filenames(self):
        filenames = self.select_folder()
        return filenames[:-2], filenames[-1]

    def select_folder(self):
        d = filedialog.askdirectory(initialdir="./videos")
        files = os.listdir(d)
        for i, file in enumerate(copy(files)):
            files[i] = os.path.join(d, file)
        files.sort(key=self.sort_files)
        return files

    @staticmethod
    def sort_files(s):
        return len(s), s
