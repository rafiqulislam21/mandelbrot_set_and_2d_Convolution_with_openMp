#include <iostream>
#include <fstream>
#include <vector>
#include <tuple>
#include <time.h>
#include <cmath>
#include <complex>
#include <chrono>
#include <omp.h>

#include "a2-helpers.hpp"

using namespace std;
//---------------------omp taskloop version---------------------------------
int num_of_thread_used = 1;
// A set of random gradients, adjusted for this mandelbrot algorithm
vector<gradient> gradients = {
    gradient({0, 0, 0}, {76, 57, 125}, 0.0, 0.010, 2000),
    gradient({76, 57, 125}, {255, 255, 255}, 0.010, 0.020, 2000),
    gradient({255, 255, 255}, {0, 0, 0}, 0.020, 0.050, 2000),
    gradient({0, 0, 0}, {0, 0, 0}, 0.050, 1.0, 2000)};

// Test if point c belongs to the Mandelbrot set
bool mandelbrot_kernel(complex<double> c, vector<int> &pixel)
{

    int max_iterations = 2048, iteration = 0;
    complex<double> z(0, 0);
    omp_set_num_threads(num_of_thread_used);

    while (abs(z) <= 4 && (iteration < max_iterations))
    {
        /* Threads update the shared counter by turns */
        z = (z * z + c);
        iteration++;

    }

    // now the computation of the color gradient and interpolation
    double length = sqrt(z.real() * z.real() + z.imag() * z.imag());
    long double m = (iteration + 1 - log(length) / log(2.0));
    double q = m / (double)max_iterations;

    q = iteration + 1 - log(log(length)) / log(2.0);
    q /= max_iterations;

    colorize(pixel, q, iteration, gradients);

    return (iteration < max_iterations);
}

/**
 * Compute the Mandelbrot set for each pixel of a given image.
 * Image is the Image data structure for storing RGB image
 * The default value for ratio is 0.15.
 *
 * @param[inout] image
 * @param[in] ratio
 *
*/
int mandelbrot(Image &image, double ratio = 0.15)
{
    int i, j;
    int h = image.height;
    int w = image.width;
    int channels = image.channels;
    ratio /= 10.0;

    int pixels_inside=0;

    // pixel to be passed to the mandelbrot function
    vector<int> pixel = {0, 0, 0}; // red, green, blue (each range 0-255)
    complex<double> c;

    //set the number of thread for parallel executation (all version)
    omp_set_num_threads(num_of_thread_used);
    //used in only task version------------------------------------------------------------------------------------
    //#pragma omp parallel default(none) private(i,j,pixel,c) shared(h, w, channels, ratio, image, pixels_inside)
    //#pragma omp single
    //used in only task version------------------------------------------------------------------------------------

    //used in openmp parallel for version--------------------------------------------------------------------------
    //#pragma omp parallel for schedule(dynamic) default(none) private(i,j,pixel,c) shared(h, w, channels, ratio, image) reduction (+:pixels_inside) collapse(2)
    //used in task openmp parallel for version---------------------------------------------------------------------

    //used in tasloop version--------------------------------------------------------------------------------------
    #pragma omp parallel default(none) private(i, j, pixel, c) shared(h, w, channels, ratio, image, pixels_inside)
    #pragma omp single
    #pragma omp taskloop collapse(2) num_tasks(omp_get_num_threads()*40) grainsize(h*40) nogroup
    //used in tasloop version--------------------------------------------------------------------------------------
    for (j = 0; j < h; j++)
    {
        //used in only task version--------------------------------------------------------------------------------
        //#pragma omp task
        for (i = 0; i < w; i++)
        {
            double dx = (double)i / (w)*ratio - 1.10;
            double dy = (double)j / (h)*0.1 - 0.35;

            c = complex<double>(dx, dy);
            if (mandelbrot_kernel(c, pixel)) // the actual mandelbrot kernel
                {
                    //used in task and taskloop both version--------------------------------------------------------
                    #pragma omp atomic
                    pixels_inside++;
                }
            // apply to the image
            for (int ch = 0; ch < channels; ch++)
                image(ch, j, i) = pixel[ch];

        }
    }

    return pixels_inside;
}

/**
 * 2D Convolution
 * src is the source Image to which we apply the filter.
 * Resulting image is saved in dst. The size of the kernel is
 * given with kernel_width (must be odd number). Sigma represents
 * the standard deviation of the filter. The number of iterations
 * is given with the nstep (default=1)
 *
 * @param[in] src
 * @param[out] dst
 * @param[in] kernel_width
 * @param[in] sigma
 * @param[in] nsteps
 *
*/
void convolution_2d(Image &src, Image &dst, int kernel_width, double sigma, int nsteps=1)
{
    int h = src.height;
    int w = src.width;
    int channels = src.channels;

    std::vector<std::vector<double>> kernel = get_2d_kernel(kernel_width, kernel_width, sigma);

    int displ = (kernel.size() / 2); // height==width!
    for (int step = 0; step < nsteps; step++)
    {

        for (int ch = 0; ch < channels; ch++)
        {
            //set the number of thread for parallel executation (all version)
            omp_set_num_threads(num_of_thread_used);
            //used in only task version-----------------------------------------------------------------------------
            //#pragma omp parallel default(none) shared(h, w, kernel, displ, ch, src, dst)
            //#pragma omp single
            //used in only task version-----------------------------------------------------------------------------

            //used in openmp parallel for version--------------------------------------------------------------------
            //#pragma omp parallel for default(none) shared(h, w, kernel, displ, ch, src, dst) collapse(2)
            //used in openmp parallel for version--------------------------------------------------------------------

            //used in tasloop version--------------------------------------------------------------------------------
            #pragma omp parallel default(none) shared(h, w, kernel, displ, ch, src, dst)
            #pragma omp single
            #pragma omp taskloop collapse(2) num_tasks(omp_get_num_threads()*40) grainsize(h*40) nogroup
            //used in tasloop version--------------------------------------------------------------------------------
            for (int i = 0; i < h; i++)
            {
                //used in task version-------------------------------------------------------------
                //#pragma omp task
                for (int j = 0; j < w; j++)
                {
                    double val = 0.0;

                    for (int k = -displ; k <= displ; k++)
                    {
                        for (int l = -displ; l <= displ; l++)
                        {
                            int cy = i + k;
                            int cx = j + l;
                            int src_val = 0;

                            // if it goes outside we disregard that value
                            if (cx < 0 || cx > w - 1 || cy < 0 || cy > h - 1) {
                                continue;
                            } else {
                                src_val = src(ch, cy, cx);
                            }

                            val += kernel[k + displ][l + displ] * src_val;
                        }
                    }
                    dst(ch, i, j) = (int)(val > 255 ? 255 : (val < 0 ? 0 : val));

                }
            }
        }
        //#pragma omp barrier
        if ( step < nsteps-1 ) {
            // swap references
            // we can reuse the src buffer for this example
            Image tmp = src; src = dst; dst = tmp;
        }
    }
}

void imageValidation(){

    char *img1data, *img2data;
	int *img1pixels, *img2pixels, *delta;
	int s, n = 0, difference = 0;

    //read sequential image
	ifstream img1("mandelbrot.ppm", ios::in|ios::binary|ios::ate);
    //read openmp parallel image
	ifstream img2("mandelbrot_openmp.ppm", ios::in|ios::binary|ios::ate);

	if (img1.is_open() && img2.is_open())
    {
		s = (int)img1.tellg()-0x54;
		//for image 1
		img1data = new char [s];
		img1pixels = new int [s/4];
		img1.seekg (0x54, ios::beg);
		img1.read (img1data, s);
		img1pixels = reinterpret_cast<int*>(img1data);
		img1.close();

        //for image 2
		img2data = new char [s];
		img2pixels = new int [s/4];
		img2.seekg (0x54, ios::beg);
		img2.read (img2data, s);
		img2pixels = reinterpret_cast<int*>(img2data);
		img2.close();

		delta = new int [s/4];

		for(int i=0; i<s/4; i++){
            delta[i] = abs (img1data[i]-img2data[i]);
            difference += delta[i];
        };

        if(difference == 0){
            cout<<"Image validation successful!"<<endl;
        }else{
            cout<<"Image validation failed!"<<endl;
        }
	}else{
        cout<<"Image not found"<<endl;
	}

}


int main(int argc, char **argv)
{
    int thread_used [5] = {1, 2, 4, 8, 16};

    // height and width of the output image
    // keep the height/width ratio for the same image
    int width = 1536, height = 1024;
    double ratio = width / (double)height;

    double time;
    int i, j, pixels_inside = 0;

    int channels = 3; // red, green, blue

    // Generate Mandelbrot set int this image
    Image image(channels, height, width);

    // Save the results of 2D convolution in this image
    Image filtered_image(channels, height, width);


    //using different trheads
    for(int i = 0; i < sizeof(thread_used)/sizeof(*thread_used); i++){
        num_of_thread_used = thread_used[i];
        cout << "Parallel executation using : " <<num_of_thread_used<<" threads."<<endl;
        auto t1 = chrono::high_resolution_clock::now();

        // Generate the mandelbrot set
        // Use OpenMP tasking to implement a parallel version
        pixels_inside = mandelbrot(image, ratio);

        auto t2 = chrono::high_resolution_clock::now();

        cout << "Mandelbrot time (openMp): " << chrono::duration<double>(t2 - t1).count() << endl;
        cout << "Total Mandelbrot pixels (openMp): " << pixels_inside << endl;

        // Actual 2D convolution part
        // Use OpenMP tasking to implement a parallel version
        auto t3 = chrono::high_resolution_clock::now();

        convolution_2d(image, filtered_image, 5, 0.37, 20);

        auto t4 = chrono::high_resolution_clock::now();

        cout << "Convolution time (openMp): " << chrono::duration<double>(t4 - t3).count() << endl;

        cout << "Total time (openMp): " << chrono::duration<double>((t4 - t3) + (t2-t1)).count() << endl;
        //20.9354 is time for mandelbrot and 51.0527 is time for convolution in sequential taken from ALMA after multiple times of running
        cout << "SpeedUp Mandelbrot: " << 20.9354/(chrono::duration<double>(t2 - t1).count()) << endl;
        cout << "SpeedUp Convolution: " << 51.0527/(chrono::duration<double>(t4 - t3).count()) << endl <<endl;
    }


    // save image
    std::ofstream ofs("mandelbrot_openmp.ppm", std::ofstream::out);
    ofs << "P3" << std::endl;
    ofs << width << " " << height << std::endl;
    ofs << 255 << std::endl;

    for (int j = 0; j < height; j++)
    {
        for (int i = 0; i < width; i++)
        {
            ofs << " " << filtered_image(0, j, i) << " " << filtered_image(1, j, i) << " " << filtered_image(2, j, i) << std::endl;
        }
    }
    ofs.close();

    // image validation for sequential and parallel
    imageValidation();

    return 0;
}
