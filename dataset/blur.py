import numpy as np
import matplotlib.pyplot as plt
from math import ceil


class Trajectory(object):
    def __init__(self, canvas=64, iters=2000, max_len=60, expl=None, path_to_save=None):
        """
        Generates a variety of random motion trajectories in continuous domain as in [Boracchi and Foi 2012]. Each
        trajectory consists of a complex-valued vector determining the discrete positions of a particle following a
        2-D random motion in continuous domain. The particle has an initial velocity vector which, at each iteration,
        is affected by a Gaussian perturbation and by a deterministic inertial component, directed toward the
        previous particle position. In addition, with a small probability, an impulsive (abrupt) perturbation aiming
        at inverting the particle velocity may arises, mimicking a sudden movement that occurs when the user presses
        the camera button or tries to compensate the camera shake. At each step, the velocity is normalized to
        guarantee that trajectories corresponding to equal exposures have the same length. Each perturbation (
        Gaussian, inertial, and impulsive) is ruled by its own parameter. Rectilinear Blur as in [Boracchi and Foi
        2011] can be obtained by setting anxiety to 0 (when no impulsive changes occurs
        :param canvas: size of domain where our trajectory os defined.
        :param iters: number of iterations for definition of our trajectory.
        :param max_len: maximum length of our trajectory.
        :param expl: this param helps to define probability of big shake. Recommended expl = 0.005.
        :param path_to_save: where to save if you need.
        """
        self.canvas = canvas
        self.iters = iters
        self.max_len = max_len
        if expl is None:
            self.expl = 0.1 * np.random.uniform(0, 1)
        else:
            self.expl = expl
        if path_to_save is None:
            pass
        else:
            self.path_to_save = path_to_save
        self.tot_length = None
        self.big_expl_count = None
        self.x = None

    def fit(self, show=False, save=False):
        """
        Generate motion, you can save or plot, coordinates of motion you can find in x property.
        Also you can fin properties tot_length, big_expl_count.
        :param show: default False.
        :param save: default False.
        :return: x (vector of motion).
        """
        tot_length = 0
        big_expl_count = 0
        # how to be near the previous position
        # TODO: I can change this paramether for 0.1 and make kernel at all image
        centripetal = 0.7 * np.random.uniform(0.5, 1)
        # probability of big shake
        prob_big_shake = 0
        # term determining, at each sample, the random component of the new direction
        gaussian_shake = 10 * np.random.uniform(0.5, 1)
        init_angle = 360 * np.random.uniform(0.5, 1)

        img_v0 = np.sin(np.deg2rad(init_angle))
        real_v0 = np.cos(np.deg2rad(init_angle))

        v0 = complex(real=real_v0, imag=img_v0)
        v = v0 * self.max_len / (self.iters - 1)

        if self.expl > 0:
            v = v0 * self.expl

        x = np.array([complex(real=0, imag=0)] * (self.iters))

        for t in range(0, self.iters - 1):
            if np.random.uniform() < prob_big_shake * self.expl:
                next_direction = 2 * v * (np.exp(complex(real=0, imag=np.pi + (np.random.uniform() - 0.5))))
                big_expl_count += 1
            else:
                next_direction = 0

            dv = next_direction + self.expl * (
                gaussian_shake * complex(real=np.random.randn(), imag=np.random.randn()) - centripetal * x[t]) * (
                                      self.max_len / (self.iters - 1))

            v += dv
            v = (v / float(np.abs(v))) * (self.max_len / float((self.iters - 1)))
            x[t + 1] = x[t] + v
            tot_length = tot_length + abs(x[t + 1] - x[t])

        # centere the motion
        x += complex(real=-np.min(x.real), imag=-np.min(x.imag))
        x = x - complex(real=x[0].real % 1., imag=x[0].imag % 1.) + complex(1, 1)
        x += complex(real=ceil((self.canvas - max(x.real)) / 2), imag=ceil((self.canvas - max(x.imag)) / 2))

        self.tot_length = tot_length
        self.big_expl_count = big_expl_count
        self.x = x

        if show or save:
            self.__plot_canvas(show, save)
        return self

    def __plot_canvas(self, show, save):
        if self.x is None:
            raise Exception("Please run fit() method first")
        else:
            plt.close()
            plt.plot(self.x.real, self.x.imag, '-', color='blue')

            plt.xlim((0, self.canvas))
            plt.ylim((0, self.canvas))
            if show and save:
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()



import numpy as np
from math import ceil
import math
import matplotlib.pyplot as plt
# from generate_trajectory import Trajectory


class PSF(object):
    def __init__(self, canvas=None, trajectory=None, fraction=None, path_to_save=None):
        if canvas is None:
            self.canvas = (canvas, canvas)
        else:
            self.canvas = (canvas, canvas)
        if trajectory is None:
            self.trajectory = Trajectory(canvas=canvas, expl=0.005).fit(show=False, save=False)
        else:
            self.trajectory = trajectory.x
        if fraction is None:
            self.fraction = [1/100, 1/10, 1/2, 1]
        else:
            self.fraction = fraction
        self.path_to_save = path_to_save
        self.PSFnumber = len(self.fraction)
        self.iters = len(self.trajectory)
        self.PSFs = []

    def fit(self, show=False, save=False):
        PSF = np.zeros(self.canvas)

        triangle_fun = lambda x: np.maximum(0, (1 - np.abs(x)))
        triangle_fun_prod = lambda x, y: np.multiply(triangle_fun(x), triangle_fun(y))
        for j in range(self.PSFnumber):
            if j == 0:
                prevT = 0
            else:
                prevT = self.fraction[j - 1]

            for t in range(len(self.trajectory)):
                # print(j, t)
                if (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t - 1):
                    t_proportion = 1
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t - 1):
                    t_proportion = self.fraction[j] * self.iters - (t - 1)
                elif (self.fraction[j] * self.iters >= t) and (prevT * self.iters < t):
                    t_proportion = t - (prevT * self.iters)
                elif (self.fraction[j] * self.iters >= t - 1) and (prevT * self.iters < t):
                    t_proportion = (self.fraction[j] - prevT) * self.iters
                else:
                    t_proportion = 0

                m2 = int(np.minimum(self.canvas[1] - 1, np.maximum(1, math.floor(self.trajectory[t].real))))
                M2 = int(m2 + 1)
                m1 = int(np.minimum(self.canvas[0] - 1, np.maximum(1, math.floor(self.trajectory[t].imag))))
                M1 = int(m1 + 1)

                PSF[m1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - m1
                )
                PSF[m1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - m1
                )
                PSF[M1, m2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - m2, self.trajectory[t].imag - M1
                )
                PSF[M1, M2] += t_proportion * triangle_fun_prod(
                    self.trajectory[t].real - M2, self.trajectory[t].imag - M1
                )

            self.PSFs.append(PSF / (self.iters))
        if show or save:
            self.__plot_canvas(show, save)

        return self.PSFs

    def __plot_canvas(self, show, save):
        if len(self.PSFs) == 0:
            raise Exception("Please run fit() method first.")
        else:
            plt.close()
            fig, axes = plt.subplots(1, self.PSFnumber, figsize=(10, 10))
            for i in range(self.PSFnumber):
                axes[i].imshow(self.PSFs[i], cmap='gray')
            if show and save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
                plt.show()
            elif save:
                if self.path_to_save is None:
                    raise Exception('Please create Trajectory instance with path_to_save')
                plt.savefig(self.path_to_save)
            elif show:
                plt.show()



import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy import signal
from scipy import misc
# from generate_PSF import PSF
# from generate_trajectory import Trajectory
from tqdm import tqdm

class BlurImage(object):

    def __init__(self, image_path, PSFs=None, part=None, path__to_save=None):
        """
        :param image_path: path to RGB image.
        :param PSFs: array of Kernels.
        :param part: int number of kernel to use.
        :param path__to_save: folder to save results.
        """
        if os.path.isfile(image_path):
            self.image_path = image_path
            self.original = cv2.imread(self.image_path)
            self.shape = self.original.shape
            if len(self.shape) < 3:
                raise Exception('We support only RGB images.')
        else:
            raise Exception('Not correct path to image.')
        self.path_to_save = path__to_save
        if PSFs is None:
            if self.path_to_save is None:
                self.PSFs = PSF(canvas=max(self.shape[0], self.shape[1])).fit()
            else:
                self.PSFs = PSF(canvas=max(self.shape[0], self.shape[1]), path_to_save=os.path.join(self.path_to_save, 'PSFs.png')).fit(save=True)
        else:
            self.PSFs = PSFs

        self.part = part
        self.result = []


    def blur_image(self, save=False, show=False):
        if self.part is None:
            psf = self.PSFs
        else:
            psf = [self.PSFs[self.part]]
        yN, xN, channel = self.shape
        for p in psf:
            # Ensure PSF is centered and padded to match image dimensions
            psf_padded = np.zeros((yN, xN))
            # Calculate center offset
            center_y, center_x = (yN - p.shape[0]) // 2, (xN - p.shape[1]) // 2
            # Place the PSF at the center of the padded array
            psf_padded[center_y:center_y + p.shape[0], center_x:center_x + p.shape[1]] = p

            # Normalize image
            blured = cv2.normalize(self.original, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            # Apply the PSF using convolution for each channel
            for i in range(channel):
                blured[:, :, i] = signal.fftconvolve(blured[:, :, i], psf_padded, 'same')
            # Normalize the blurred image to scale 0 to 255
            blured = cv2.normalize(blured, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            blured = cv2.cvtColor(blured, cv2.COLOR_RGB2BGR)
            self.result.append(blured)

        if show or save:
            self.__plot_canvas(show, save)


    def __plot_canvas(self, show, save):
        if len(self.result) == 0:
            raise Exception('Please run blur_image() method first.')
        else:
            plt.close()
            plt.axis('off')
            fig, axes = plt.subplots(1, len(self.result), figsize=(10, 10))
            if len(self.result) > 1:
                for i in range(len(self.result)):
                    axes[i].imshow(cv2.cvtColor(self.result[i], cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(cv2.cvtColor(self.result[0], cv2.COLOR_BGR2RGB))
            if save:
                if self.path_to_save is None:
                    raise Exception('Please specify a path to save.')
                # 处理文件路径分隔符问题
                filename = os.path.basename(self.image_path)
                for idx, res in enumerate(self.result):
                    save_path = os.path.join(self.path_to_save, filename)
                    print(f"保存模糊图像到: {save_path}")
                    cv2.imwrite(save_path, res)
            if show:
                plt.show()
                

if __name__ == '__main__':
    # 导入所需模块
    import traceback
    
    folders = ['F:/avisdrone2019/VisDrone2019-DET-train/images',
              'F:/avisdrone2019/VisDrone2019-DET-val/images',
              'F:/avisdrone2019/VisDrone2019-DET-test-dev/images',
    ]

    folder_to_saves = ['F:/VisDrone-2019-DET_blur/VisDrone2019-DET-train/images',
                      'F:/VisDrone-2019-DET_blur/VisDrone2019-DET-val/images',
                      'F:/VisDrone-2019-DET_blur/VisDrone2019-DET-test-dev/images',
    ]

    # 创建失败记录文件
    error_log = open('blur_error_log.txt', 'a')

    for index in range(3):
        folder = folders[index]
        folder_to_save = folder_to_saves[index]
        os.makedirs(folder_to_save, exist_ok=True)
        params = [0.001, 0.0012, 0.0014, 0.0016, 0.0018, 0.002]
        
        # 获取文件列表
        files = os.listdir(folder)
        print(f"处理文件夹 {folder}，共有 {len(files)} 张图片")
        
        for path in tqdm(files):
            # 检查目标文件是否已存在
            output_file = os.path.join(folder_to_save, path)
            if os.path.exists(output_file):
                print(f"跳过已存在的文件: {path}")
                continue
            
            # 限制尝试次数
            max_tries = 5
            tries = 0
            success = False
            
            while tries < max_tries and not success:
                tries += 1
                try:
                    print(f"处理: {path}，尝试 {tries}/{max_tries}")
                    expl = np.random.choice(params)
                    print(f"模糊参数: {expl}")
                    
                    # 生成轨迹和PSF
                    trajectory = Trajectory(canvas=64, max_len=100, expl=expl).fit()
                    psf = PSF(canvas=64, trajectory=trajectory).fit()
                    
                    # 应用模糊效果并保存
                    blur_img = BlurImage(os.path.join(folder, path), 
                                        PSFs=psf,
                                        path__to_save=folder_to_save, 
                                        part=np.random.choice([1, 2, 3]))
                    blur_img.blur_image(save=True)
                    
                    # 确认文件是否已保存
                    if os.path.exists(output_file):
                        print(f"已成功保存文件: {output_file}")
                        success = True
                    else:
                        print(f"警告: 文件处理完成但未找到输出文件: {output_file}")
                        
                except Exception as e:
                    print(f"错误处理 {path}: {str(e)}")
                    error_log.write(f"图片 {path} 处理失败，尝试 {tries}/{max_tries}: {str(e)}\n")
                    error_log.write(traceback.format_exc() + "\n")
                    
            if not success:
                print(f"无法处理图片 {path}，已尝试 {max_tries} 次")
                error_log.write(f"图片 {path} 处理失败，已尝试最大次数 {max_tries}\n")
    
    # 关闭错误日志
    error_log.close()