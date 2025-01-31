// Compile with     g++ -o bg_prova bg_prova.cpp `pkg-config --cflags --libs opencv`

#include <iostream>
#include <algorithm>
#include <vector>
#include <iterator>
#include <utility>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/geometries/linestring.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/adapted/boost_tuple.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>
#include <boost/geometry/geometries/adapted/boost_array.hpp>
#include <boost/geometry/geometries/adapted/boost_polygon/point.hpp>
#include <boost/geometry/geometries/register/linestring.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/multi_polygon.hpp>

#include <boost/foreach.hpp>

BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)
BOOST_GEOMETRY_REGISTER_BOOST_ARRAY_CS(cs::cartesian)
BOOST_GEOMETRY_REGISTER_BOOST_TUPLE_CS(cs::cartesian)

namespace boost {
namespace geometry {
namespace model {
    template <typename T>
    bool operator==(const d2::point_xy<T>& p1, const d2::point_xy<T>& p2) {
        return p1.x() == p2.x() && p1.y() == p2.y();
    }
}
}
}

using namespace boost::geometry;

typedef model::d2::point_xy<double> point_2d;
typedef model::linestring<point_2d> linestring_2d;
typedef model::polygon<point_2d> polygon_2d;
typedef model::box<point_2d> box_2d;

struct intersection_point{
    point_2d point;
    double distance;
    double angle;
    linestring_2d parallelogram_vertexs;
    //std::vector<int> indexs = std::vector<int>(2);
};

struct rect{
    linestring_2d vrtxs;
    point_2d centr;
    double area;
};

struct SDF_poly{
    double SDFvalue;
    int index;
};

struct maxSDF_polyline{
    std::vector<linestring_2d> polyline = {};
    std::vector<size_t> polyline_indexs;
};

// Parameters
const bool flag_save_img = false;
const bool flag_print_file = false;
const bool flag_save_lenmask = false;
const bool flag_save_anglemask = false;
const bool flag_save_rects = false;
const std::string n_per = "4";
const std::string strng_sweep_angle = "0";
const std::string strng_angle_step = "0";
//const std::string strng_tol = "10°";
const double sweep_angle = 0;
const double angle_step = 0;
//const double tol = 10;
const double window_factor = 0.04;

const int scale_img = 15;
int N = 0;

const double dist_tol = 0.5;
const double len_tol = 1;
const double angle_tol = 5;
const int nbeams_tol = 1;
const std::string nbeams_string = "1";

std::vector<std::vector<polygon_2d>> sliced_polys;


inline double rad2deg(double alpha){
    return alpha*180/M_PI;
}

inline double deg2rad(double alpha){
    return alpha*M_PI/180;
}

void calculateSDF(const polygon_2d& polygon);

void create_polygon(polygon_2d& polygon){
    std::ifstream file;
    file.open("../dataset_perimetri/" + n_per + "/lista_punti.txt", std::ios::in);

    std::vector<double> tmp;
    std::string row;

    while(std::getline(file, row)){

        std::stringstream ss(row);
        std::string value;

        while (std::getline(ss, value, ',')) {
            double elem = std::stod(value);
            tmp.push_back(elem);
        }

    }

    double x;
    for(int i = 0; i < tmp.size(); i++){
        if(i % 2 == 0){
            x = tmp[i];
        }
        else{
            append(polygon, make<point_2d>(x, tmp[i]));
        }
    }

    file.close();
    correct(polygon);
}

void calculateLinearFit(const linestring_2d& pts, double& m, double& q){

    if (pts.size() < 2) {
        throw std::invalid_argument("Sono necessari almeno due punti per calcolare la retta interpolante.");
    }

    double sum_x = 0;
    double sum_y = 0;
    double sum_xy = 0;
    double sum_xx = 0;

    for(int i = 0; i < pts.size(); i++){
        sum_x += pts[i].x();
        sum_y += pts[i].y();
        sum_xx += pts[i].x() * pts[i].x();
        sum_xy += pts[i].x() * pts[i].y();
    }

    double mean_x = sum_x / pts.size();
    double mean_y = sum_y / pts.size();

    double num = sum_xy - pts.size() * mean_x * mean_y;
    double den = sum_xx - pts.size() * mean_x * mean_x;

    if (den == 0) {
        throw std::runtime_error("Impossibile calcolare la retta: i punti potrebbero essere allineati verticalmente.");
    }

    m = num / den;
    q = mean_y - m * mean_x;
}

void visualize_masks(const polygon_2d& polygon, const std::vector<linestring_2d>& normal_beams, const std::vector<int>& index_normalbeams, const std::vector<linestring_2d>& polylines, const std::vector<double>& normalbeams_len, const std::vector<double>& normalbeams_angle, const std::vector<intersection_point>& vec_interpoints, const std::vector<double>& SDFvalues_lines){
    
    const double max_distance = *std::max_element(normalbeams_len.begin(), normalbeams_len.end());
    const double max_angle = *std::max_element(normalbeams_angle.begin(), normalbeams_angle.end());
    const double max_SDFvalue = *std::max_element(SDFvalues_lines.begin(), SDFvalues_lines.end());

    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        x_poly.push_back(exterior_ring(polygon)[i].x());
        y_poly.push_back(exterior_ring(polygon)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end());
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end());

    point_2d bbox_center((max_x + min_x)/2, (max_y + min_y)/2);
    double bbox_dims[] = {(max_x - min_x), (max_y - min_y)};

    const int img_scalefactor = (int)std::max(max_x - min_x, max_y - min_y)*100 > 1000 ? scale_img : 100;

    cv::Mat gray_image_len((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));
    cv::Mat gray_image_angle((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));
    cv::Mat gray_image_perimeter((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    cv::Mat interpoint_polys((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    std::vector<cv::Point> poly_mask;

    //std::cout << "[" << (int)(1.2 * img_scalefactor * bbox_dims[0]) << ", " << (int)(1.2 * img_scalefactor * bbox_dims[1]) << "]" << std::endl;

    // drawing perimeter
    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        poly_mask.push_back(start);
        cv::line(gray_image_perimeter, start, end, cv::Scalar(170), 1);
        cv::line(interpoint_polys, start, end, cv::Scalar(128), 1);
    }

    // drawing masks
    for(int i = 0; i < normal_beams.size(); i++){
        double start_x = img_scalefactor * (normal_beams[i].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (normal_beams[i].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (normal_beams[i].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (normal_beams[i].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image_len, start, end, cv::Scalar((normalbeams_len[i]*255/max_distance)), (int)(img_scalefactor * distance(polylines[index_normalbeams[i]].front(), polylines[index_normalbeams[i]].back())));
        cv::line(gray_image_angle, start, end, cv::Scalar((normalbeams_angle[i]*(255 - 50)/max_angle + 50)), (int)(img_scalefactor * distance(polylines[index_normalbeams[i]].front(), polylines[index_normalbeams[i]].back())));
        //cv::line(gray_image_len, start, end, cv::Scalar((normalbeams_len[i]*255/max_distance)), 1);
        //cv::line(gray_image_angle, start, end, cv::Scalar((normalbeams_angle[i]*(255 - 50)/max_angle + 50)), 1);
    }

    // handling visualization of intersection points
    for(int i = 0; i < vec_interpoints.size(); i++){
        double y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (vec_interpoints[i].point.y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double x = img_scalefactor * (vec_interpoints[i].point.x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        if(gray_image_len.at<uchar>(y, x) < (int)(vec_interpoints[i].distance * 255 / max_distance)){
            
            std::cout << "[" << i << "]: " << "(" << x << ", " << y << ")" << std::endl;
            
            cv::Point p_mm_img(img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[0].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[0].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);
            cv::Point p_mp_img(img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);
            cv::Point p_pp_img(img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[2].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[2].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);
            cv::Point p_pm_img(img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[3].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (vec_interpoints[i].parallelogram_vertexs[3].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);

            std::vector<cv::Point> parallelogram_vertexs_img = {p_mm_img, p_mp_img, p_pp_img, p_pm_img};

            //interpoint_polys.at<uchar>(p_mm_img.y, p_mm_img.x) = 255;
            //interpoint_polys.at<uchar>(p_mp_img.y, p_mp_img.x) = 255;
            //interpoint_polys.at<uchar>(p_pp_img.y, p_pp_img.x) = 255;
            //interpoint_polys.at<uchar>(p_pm_img.y, p_pm_img.x) = 255;

            //cv::drawMarker(interpoint_polys, p_mm_img, cv::Scalar(255), cv::MARKER_TILTED_CROSS);
            //cv::drawMarker(interpoint_polys, p_mp_img, cv::Scalar(255), cv::MARKER_TILTED_CROSS);
            //cv::drawMarker(interpoint_polys, p_pp_img, cv::Scalar(255), cv::MARKER_TILTED_CROSS);
            //cv::drawMarker(interpoint_polys, p_pm_img, cv::Scalar(255), cv::MARKER_TILTED_CROSS);

            std::cout << "\tp_mm: " << "(" << p_mm_img.x << ", " << p_mm_img.y << ")";// << std::endl;
            std::cout << "\tp_mp: " << "(" << p_mp_img.x << ", " << p_mp_img.y << ")";// << std::endl;
            std::cout << "\tp_pp: " << "(" << p_pp_img.x << ", " << p_pp_img.y << ")";// << std::endl;
            std::cout << "\tp_pm: " << "(" << p_pm_img.x << ", " << p_pm_img.y << ")";// << std::endl;
            std::cout << std::endl << "\tDistance pixel: " << static_cast<int>(gray_image_len.at<uchar>(y, x)) << "\tDistance candidate: " << (int)(vec_interpoints[i].distance * 255 / max_distance);
            std::cout << std::endl << "\tAngle pixel: " << static_cast<int>(gray_image_angle.at<uchar>(y, x)) << "\tAngle candidate: " << (int)(normalbeams_angle[i]*(255 - 50)/max_angle + 50);
            std::cout << std::endl;

            cv::fillConvexPoly(gray_image_len, parallelogram_vertexs_img, cv::Scalar(vec_interpoints[i].distance * 255 / max_distance));
            cv::fillConvexPoly(gray_image_angle, parallelogram_vertexs_img, cv::Scalar(normalbeams_angle[i]*(255 - 50)/max_angle + 50));
            cv::fillConvexPoly(interpoint_polys, parallelogram_vertexs_img, cv::Scalar(255));
            //gray_image_len.at<uchar>(y, x) = vec_interpoints[i].distance * 255 / max_distance;
            //gray_image_angle.at<uchar>(y, x) = vec_interpoints[i].angle * 255 / max_angle;
        }
    }

    cv::imshow("Parallelograms points", interpoint_polys);

    //std::cout << vec_interpoints.size() << std::endl;

    cv::Mat sum_img_len, sum_img_angle;
    cv::add(gray_image_len, gray_image_perimeter, sum_img_len);
    cv::add(gray_image_angle, gray_image_perimeter, sum_img_angle);

    cv::Mat heatmap_len, heatmap_angle;
    cv::applyColorMap(sum_img_len, heatmap_len, cv::COLORMAP_JET);
    cv::applyColorMap(sum_img_angle, heatmap_angle, cv::COLORMAP_JET);

    std::string strng_pathfile_len = "lenmask_imgs/lenmask_" + n_per + "_notol.jpg";
    std::string strng_pathfile_angle = "anglemask_imgs/anglemask_" + n_per + "_notol.jpg";

    if(flag_save_lenmask)
        cv::imwrite(strng_pathfile_len, heatmap_len);
    cv::imshow("Lenght mask", heatmap_len);

    if(flag_save_anglemask)
        cv::imwrite(strng_pathfile_angle, heatmap_angle);
    cv::imshow("Angle mask", heatmap_angle);

}

void visualize_masks_v2(const polygon_2d& polygon, const std::vector<linestring_2d>& normal_beams, const std::vector<int>& index_normalbeams, const std::vector<linestring_2d>& polylines, const std::vector<double>& normalbeams_len, const std::vector<double>& normalbeams_angle){

    const double max_distance = *std::max_element(normalbeams_len.begin(), normalbeams_len.end());
    const double max_angle = *std::max_element(normalbeams_angle.begin(), normalbeams_angle.end());
    //const double max_SDFvalue = *std::max_element(SDFvalues_lines.begin(), SDFvalues_lines.end());

    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        x_poly.push_back(exterior_ring(polygon)[i].x());
        y_poly.push_back(exterior_ring(polygon)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end());
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end());

    point_2d bbox_center((max_x + min_x)/2, (max_y + min_y)/2);
    double bbox_dims[] = {(max_x - min_x), (max_y - min_y)};

    const int img_scalefactor = (int)std::max(max_x - min_x, max_y - min_y)*100 > 1000 ? scale_img : 100;

    cv::Mat gray_image_len((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));
    cv::Mat gray_image_angle((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));
    cv::Mat gray_image_perimeter((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));
    

    std::vector<double> lenbeams_sorted = normalbeams_len;
    std::sort(lenbeams_sorted.begin(), lenbeams_sorted.end());

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image_perimeter, start, end, cv::Scalar(170), 1);
    }

    for(int j = 0; j < lenbeams_sorted.size(); j++){
        for(int i = 0; i < normal_beams.size(); i++){
            if(normalbeams_len[i] == lenbeams_sorted[j]){
                double start_x = img_scalefactor * (normal_beams[i].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (normal_beams[i].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                double end_x = img_scalefactor * (normal_beams[i].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (normal_beams[i].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                cv::Point start(start_x, start_y);
                cv::Point end(end_x, end_y);
                cv::line(gray_image_len, start, end, cv::Scalar((normalbeams_len[i]*255/max_distance)), (int)(img_scalefactor * distance(polylines[index_normalbeams[i]].front(), polylines[index_normalbeams[i]].back())));
                cv::line(gray_image_angle, start, end, cv::Scalar((normalbeams_angle[i]*(255 - 50)/max_angle + 50)), (int)(img_scalefactor * distance(polylines[index_normalbeams[i]].front(), polylines[index_normalbeams[i]].back())));
                //std::cout << std::endl << "Angle beam (visual) [" << j << "]: " << normalbeams_angle[j] << "\tLength beam (visual) [" << j << "]: " << normalbeams_len[j];
            }
        }
    }


    cv::Mat sum_img_len, sum_img_angle;
    cv::add(gray_image_len, gray_image_perimeter, sum_img_len);
    cv::add(gray_image_angle, gray_image_perimeter, sum_img_angle);

    cv::Mat heatmap_len, heatmap_angle;
    cv::applyColorMap(sum_img_len, heatmap_len, cv::COLORMAP_JET);
    cv::applyColorMap(sum_img_angle, heatmap_angle, cv::COLORMAP_JET);

    std::string strng_pathfile_len = "lenmask_thickbeams_imgs/lenmask_" + n_per + "_notol.jpg";
    std::string strng_pathfile_angle = "anglemask_thickbeams_imgs/anglemask_" + n_per + "_notol.jpg";

    if(flag_save_lenmask)
        cv::imwrite(strng_pathfile_len, heatmap_len);
    cv::imshow("Lenght mask", heatmap_len);

    if(flag_save_anglemask)
        cv::imwrite(strng_pathfile_angle, heatmap_angle);
    cv::imshow("Angle mask", heatmap_angle);

}

void visualize_rects(const polygon_2d& polygon, const std::vector<int>& index_normalbeams, const std::vector<rect>& rects, const linestring_2d& mid_pts, const std::vector<double>& normalbeams_len, const std::vector<double>& normalbeams_angle){
    
    const double max_distance = *std::max_element(normalbeams_len.begin(), normalbeams_len.end());
    const double max_angle = *std::max_element(normalbeams_angle.begin(), normalbeams_angle.end());
    //const double max_SDFvalue = *std::max_element(SDFvalues_lines.begin(), SDFvalues_lines.end());

    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        x_poly.push_back(exterior_ring(polygon)[i].x());
        y_poly.push_back(exterior_ring(polygon)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end());
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end());

    point_2d bbox_center((max_x + min_x)/2, (max_y + min_y)/2);
    double bbox_dims[] = {(max_x - min_x), (max_y - min_y)};

    const int img_scalefactor = (int)std::max(max_x - min_x, max_y - min_y)*100 > 1000 ? scale_img : 100;

    cv::Mat img_rects((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));
    cv::Mat gray_image_perimeter((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image_perimeter, start, end, cv::Scalar(170), 1);
    }

    for(int i = 0; i < mid_pts.size(); i++){
        double x = img_scalefactor * (mid_pts[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (mid_pts[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        img_rects.at<uchar>(y, x) = 100;
    }

    std::vector<double> areas_rects;
    for(int i = 0; i < rects.size(); i++)
        areas_rects.push_back(rects[i].area);

    std::vector<double> areasrects_sorted = areas_rects;
    std::sort(areasrects_sorted.begin(), areasrects_sorted.end(), std::greater<double>());
    //std::sort(areasrects_sorted.begin(), areasrects_sorted.end());

    std::vector<size_t> index_areasrects;
    for(int i = 0; i < areasrects_sorted.size(); i++)
        for(int j = 0; j < areas_rects.size(); j++)
            if(areas_rects[j] == areasrects_sorted[i])
                index_areasrects.push_back(j);

    //for(int i = 0; i < index_areasrects.size(); i++) std::cout << std::endl << "[" << i << "]: " << index_areasrects[i];

    for(int i = 0; i < rects.size() * 0.2; i++){
    //int i = 4;{

        /*for(int j = 0; j < rects[i].midpts.size(); j++){
            double x = img_scalefactor * (rects[i].midpts[j].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (rects[i].midpts[j].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            img_rects.at<uchar>(y, x) = 100;
        }*/
        
       //std::cout << i << std::endl;

        cv::Point p_mm_img(img_scalefactor * (rects[index_areasrects[i]].vrtxs[0].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (rects[index_areasrects[i]].vrtxs[0].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);
        cv::Point p_mp_img(img_scalefactor * (rects[index_areasrects[i]].vrtxs[1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (rects[index_areasrects[i]].vrtxs[1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);
        cv::Point p_pp_img(img_scalefactor * (rects[index_areasrects[i]].vrtxs[2].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (rects[index_areasrects[i]].vrtxs[2].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);
        cv::Point p_pm_img(img_scalefactor * (rects[index_areasrects[i]].vrtxs[3].x() + (1.2*bbox_dims[0]/2 - bbox_center.x())) + 0.5, (1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (rects[index_areasrects[i]].vrtxs[3].y() + (1.2*bbox_dims[1]/2 - bbox_center.y())) + 0.5);
        std::vector<cv::Point> vertexes = {p_mm_img, p_mp_img, p_pp_img, p_pm_img};
        
        cv::polylines(img_rects, vertexes, true, cv::Scalar(i * (255 - 70)/(rects.size()*0.2) + 70), 2);
    }

    cv::Mat sum_img;
    cv::add(img_rects, gray_image_perimeter, sum_img);

    cv::Mat heatmap;
    cv::applyColorMap(sum_img, heatmap, cv::COLORMAP_JET);

    std::string strng_pathfile_rects = "rects_imgs/nbeams" + nbeams_string + "/rects_" + n_per + "_notol.jpg";

    if(flag_save_rects)
        cv::imwrite(strng_pathfile_rects, heatmap);
    cv::imshow("Rects", heatmap);

}

void smoothing_SDF(std::vector<double>& SDFvalues_lines, std::vector<linestring_2d>& polylines, std::vector<double>& normals_angles){

    std::vector<double>::iterator max_SDFvalue_index = std::max_element(SDFvalues_lines.begin(), SDFvalues_lines.end());
    //SDFvalues_lines[std::distance(SDFvalues_lines.begin(), max_SDFvalue_index)] = 0;
    SDFvalues_lines.erase(SDFvalues_lines.begin() + std::distance(SDFvalues_lines.begin(), max_SDFvalue_index));
    polylines.erase(polylines.begin() + std::distance(SDFvalues_lines.begin(), max_SDFvalue_index));
    normals_angles.erase(normals_angles.begin() + std::distance(SDFvalues_lines.begin(), max_SDFvalue_index));

    max_SDFvalue_index = std::max_element(SDFvalues_lines.begin(), SDFvalues_lines.end());
    //SDFvalues_lines[std::distance(SDFvalues_lines.begin(), max_SDFvalue_index)] = 0;
    SDFvalues_lines.erase(SDFvalues_lines.begin() + std::distance(SDFvalues_lines.begin(), max_SDFvalue_index));

    std::vector<double> SDF_copy = SDFvalues_lines;
    std::vector<double> SDF_sorted = SDFvalues_lines;
    std::vector<double> SDF_mean(SDFvalues_lines.size(), 0);
    std::sort(SDF_sorted.begin(), SDF_sorted.end(), std::greater<double>());
    for(int i = 0; i < SDF_sorted.size() - 7; i++){
        for(int j = 0; j < 5; j++){
            SDF_mean[i] += SDF_sorted[i+j];
        }
        SDF_mean[i] /= 5;
        for(int j = 0; j < SDFvalues_lines.size(); j++){
            if(SDF_sorted[i] == SDFvalues_lines[j])
                SDF_copy[j] = SDF_mean[i];
        }
    }

    SDFvalues_lines = SDF_copy;

    //std::sort(SDF_sorted.begin(), SDF_sorted.end());
    //for(int i = 0; i < SDF_sorted.size(); i++) std::cout << "[" << i << "]:\t" << SDF_sorted[i] << std::endl;

}

void moving_average(std::vector<double>& vector, const int& n_values){
    const int window = vector.size() * window_factor;
    //std::cout << window << std::endl;
    //std::vector<double> vect = vector;

    for(int i = 0; i < vector.size(); i++){
        double mean_num = vector[i];
        for(int j = 1; j <= window/2; j++){
            if(i + j >= vector.size()){
                mean_num += vector[abs((int)vector.size() - (i + j))];
            }
            else{
                mean_num += vector[i + j];
            }
            j *= -1;
            if(i + j < 0){
                mean_num += vector[abs((int)vector.size() + i + j)];
            }
            else{
                mean_num += vector[i + j];
            }
            j *= -1;
        }
        vector[i] = std::round(mean_num / (window % 2 == 0 ? window + 1 : window) * n_values) / (double)n_values;
    }

    //vector = vect;
}

void visualize_polylines(const polygon_2d& polygon, const std::vector<maxSDF_polyline>& maxSDF_polylines){
    
    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        x_poly.push_back(exterior_ring(polygon)[i].x());
        y_poly.push_back(exterior_ring(polygon)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end());
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end());

    point_2d bbox_center((max_x + min_x)/2, (max_y + min_y)/2);
    double bbox_dims[] = {(max_x - min_x), (max_y - min_y)};

    const int img_scalefactor = (int)std::max(max_x - min_x, max_y - min_y)*100 > 1000 ? scale_img : 100;

    cv::Mat gray_image((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image, start, end, cv::Scalar(150), 1);
    }

    for(int i = 0; i < maxSDF_polylines.size(); i++){
        for(int j = 0; j < maxSDF_polylines[i].polyline.size(); j++){
            double start_x = img_scalefactor * (maxSDF_polylines[i].polyline[j].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (maxSDF_polylines[i].polyline[j].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            double end_x = img_scalefactor * (maxSDF_polylines[i].polyline[j].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (maxSDF_polylines[i].polyline[j].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            cv::Point start(start_x, start_y);
            cv::Point end(end_x, end_y);
            cv::line(gray_image, start, end, cv::Scalar(255), 1);
        }
    }

    std::string strng_polylines_file = "SDFpolylines/per_" + n_per + "_gen0.jpg";
    //cv::imwrite(strng_polylines_file, gray_image);
    cv::imshow("Max SDF values", gray_image);
}

void visualize_childpolys(const polygon_2d& polygon, const std::vector<polygon_2d>& cut_polys){
    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        x_poly.push_back(exterior_ring(polygon)[i].x());
        y_poly.push_back(exterior_ring(polygon)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end());
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end());

    point_2d bbox_center((max_x + min_x)/2, (max_y + min_y)/2);
    double bbox_dims[] = {(max_x - min_x), (max_y - min_y)};

    const int img_scalefactor = (int)std::max(max_x - min_x, max_y - min_y)*100 > 1000 ? scale_img : 80;

    cv::Mat gray_image((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image, start, end, cv::Scalar(150), 1);
    }

    for(int j = 0; j < cut_polys.size(); j++){
        for(int i = 0; i < num_points(cut_polys[j]) - 1; i++){
            double start_x = img_scalefactor * (exterior_ring(cut_polys[j])[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(cut_polys[j])[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            double end_x = img_scalefactor * (exterior_ring(cut_polys[j])[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(cut_polys[j])[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            cv::Point start(start_x, start_y);
            cv::Point end(end_x, end_y);
            cv::line(gray_image, start, end, cv::Scalar(255), 1);
        }
    }

    std::string strng_childpolys_file = "SDFchildpolys/per_" + n_per + "_gen0.jpg";
    //cv::imwrite(strng_childpolys_file, gray_image);
    std::string strng_name = "Child polys gen #" + std::to_string(N+1);
    cv::imshow(strng_name, gray_image);
    /*for(int i = 0; i < cut_polys.size(); i++){
        std::string strng_name = "Child poly #" + std::to_string(i);
        cv::imshow(strng_name, gray_image);
    }*/
}

int get_paired_polylines(const std::vector<std::vector<int>>& paired_indexs, const int& j){
    for(int k = 0; k < paired_indexs.size(); k++){
        if(paired_indexs[k].front() == j){
            return paired_indexs[k].back();
        }
        else if(paired_indexs[k].back() == j){
            return paired_indexs[k].front();
        }
    }
}

bool checkif_polylineispaired(const std::vector<std::vector<int>>& paired_indexs, const int& j){

    for(int i = 0; i < paired_indexs.size(); i++){
        if(paired_indexs[i].front() == j || paired_indexs[i].back() == j){
            return true;
        }
    }

    return false;

}

bool checkif_pairisnotused(const std::vector<std::vector<int>>& paired_indexs, const int& j, const std::vector<std::vector<int>>& used_pair){

    if(used_pair.empty()) return true;

    for(int k = 0; k < used_pair.size(); k++)
        for(int i = 0; i < paired_indexs.size(); i++){
            if(used_pair[k].front() == paired_indexs[i].front() && used_pair[k].back() == paired_indexs[i].back()){
                continue;
            }
            else if(used_pair[k].front() == paired_indexs[i].front() || used_pair[k].back() == paired_indexs[i].front() || used_pair[k].front() == paired_indexs[i].back() || used_pair[k].back() == paired_indexs[i].back()){
                return false;
            }
        }

    return true;
}

void distinct_pairs(std::vector<std::vector<int>>& paired_indexs){

    if(paired_indexs.size() == 1) return;

    for(int j = 0; j < paired_indexs.size(); j++)
        for(int i = j + 1; i < paired_indexs.size(); i++){
            if(paired_indexs[j].front() == paired_indexs[i].front() || paired_indexs[j].back() == paired_indexs[i].front() || paired_indexs[j].front() == paired_indexs[i].back() || paired_indexs[j].back() == paired_indexs[i].back()){
                paired_indexs.erase(paired_indexs.begin() + i + j);
                i--;
            }
        }

}

void adjust_length_polylines(polygon_2d& polygon){
    std::vector<double> len_polylines(num_points(polygon) - 1);

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double len = distance(exterior_ring(polygon)[i], exterior_ring(polygon)[i+1]);
        if(len > 0.2){
            double m = (exterior_ring(polygon)[i+1].y() - exterior_ring(polygon)[i].y())/(exterior_ring(polygon)[i+1].x() - exterior_ring(polygon)[i].x());
            double normdir_factor = 1./sqrt(1 + m*m);
            for(int j = 1; j < len/0.1; j++){
                point_2d new_pt;
                if(exterior_ring(polygon)[i].x() < exterior_ring(polygon)[i+1].x()){
                    new_pt = make<point_2d>(exterior_ring(polygon)[i].x() + j * normdir_factor * 0.1, exterior_ring(polygon)[i].y() + j * normdir_factor * m * 0.1);
                }
                else{
                    new_pt = make<point_2d>(exterior_ring(polygon)[i].x() - j * normdir_factor * 0.1, exterior_ring(polygon)[i].y() - j * normdir_factor * m * 0.1);
                }
                exterior_ring(polygon).insert(exterior_ring(polygon).begin() + i + j, new_pt);
            }
        }
    }
    correct(polygon);
}

void slicing_polys(const polygon_2d& polygon, std::vector<double>& normalized_SDF, const std::vector<linestring_2d>& polylines){
    const double maxSDF = *std::max_element(normalized_SDF.begin(), normalized_SDF.end());
    const double minSDF = *std::min_element(normalized_SDF.begin(), normalized_SDF.end());
    const double alpha = 4;
    const int n_values = 1;

    for(int i = 0; i < normalized_SDF.size(); i++){
        normalized_SDF[i] = std::log((normalized_SDF[i] - minSDF)/(maxSDF - minSDF) * alpha + 1) / std::log(alpha + 1);
        normalized_SDF[i] = std::round(normalized_SDF[i] * n_values) / (double)n_values;
        //if(N == 2)  std::cout << "NormSDF[" << i << "]:\t" << normalized_SDF[i] << std::endl;
    }

    moving_average(normalized_SDF, n_values);

    std::vector<size_t> maxSDF_indexs;

    for(int i = 0; i < normalized_SDF.size(); i++){
        if(normalized_SDF[i] == 1){
            maxSDF_indexs.push_back(i);
        }
    }

    std::vector<maxSDF_polyline> maxSDF_polylines;

    int k = 0;
    std::vector<linestring_2d> tmp_polyline;
    std::vector<size_t> tmp_indexs;
    maxSDF_polyline tmp_maxSDF;
    std::cout << std::endl;
    for(int i = 0; i <= maxSDF_indexs.size(); i++){
        if(k != 0 && maxSDF_indexs[i] - maxSDF_indexs[i-1] != 1){
            k = 0;
            tmp_maxSDF.polyline = tmp_polyline;
            tmp_maxSDF.polyline_indexs = tmp_indexs;
            maxSDF_polylines.push_back(tmp_maxSDF);
            tmp_polyline.clear();
            tmp_indexs.clear();
        }
        if(i == maxSDF_indexs.size()) break;
        tmp_indexs.push_back(maxSDF_indexs[i]);
        //std::cout << "[" << k << "]:\t" << tmp_indexs[k] << std::endl;
        tmp_polyline.push_back(polylines[maxSDF_indexs[i]]);
        k++;
    }

    /*std::cout << std::endl;
    for(int i = 0; i < maxSDF_polylines.size(); i++){
        std::cout << "Polyline [" << i << "]\t" << std::endl;
        for(int j = 0; j < maxSDF_polylines[i].polyline_indexs.size(); j++){
            std::cout << "\tIndex [" << j << "]: " << maxSDF_polylines[i].polyline_indexs[j] << std::endl;
        }
    }*/

    visualize_polylines(polygon, maxSDF_polylines);
    //if(N == 1) return;

    std::vector<polygon_2d> cut_polys;
    std::vector<std::vector<int>> paired_indexs;
    for(int i = 0; i < maxSDF_polylines.size(); i++){
    //int i = 0;{
        polygon_2d newtry_poly;
        for(int k = 0; k < maxSDF_polylines[i].polyline.size(); k++){
            newtry_poly.outer().push_back(maxSDF_polylines[i].polyline[k].front());
            newtry_poly.outer().push_back(maxSDF_polylines[i].polyline[k].back());
        }
        for(int j = i; j < maxSDF_polylines.size(); j++){
        //int j = 1;{
            if(j != i || maxSDF_polylines.size() == 1){
                for(int k = 0; k < maxSDF_polylines[j].polyline.size(); k++){
                    newtry_poly.outer().push_back(maxSDF_polylines[j].polyline[k].front());
                    newtry_poly.outer().push_back(maxSDF_polylines[j].polyline[k].back());
                }

                newtry_poly.outer().erase(std::unique(newtry_poly.outer().begin(), newtry_poly.outer().end()), newtry_poly.outer().end());
                correct(newtry_poly);

                std::vector<polygon_2d> inter;
                intersection(polygon, newtry_poly, inter);

                if(inter.size() != 0){
                    correct(inter.front());
                    de9im::mask mask("T*F**FFF*");
                    if(relate(newtry_poly, inter.front(), mask)){
                        std::vector<int> tmp;
                        cut_polys.push_back(newtry_poly);
                        std::cout << "Pair [" << i << ", " << j << "]" << std::endl;
                        tmp.push_back(i);
                        tmp.push_back(j);
                        paired_indexs.push_back(tmp);
                    }
                }

                newtry_poly.outer().erase(newtry_poly.outer().begin() + maxSDF_polylines[i].polyline.size(), newtry_poly.outer().end());

            }
        }
    }

    //visualize_childpolys(polygon, cut_polys);

    if(paired_indexs.empty()) return;

    distinct_pairs(paired_indexs);

    std::cout << std::endl;
    std::cout << "Post erase" << std::endl;
    for(int i = 0; i < paired_indexs.size(); i++){
        std::cout << "\tPair [" << paired_indexs[i].front() << ", " << paired_indexs[i].back() << "]" << std::endl;
    }

    //std::cout << std::endl;
    //std::cout << "Generazione " << N << std::endl;

    std::vector<polygon_2d> child_polys;
    std::vector<linestring_2d> polylines_copy = polylines;
    std::vector<size_t> polylines_indexs;
    
    for(int i = 0; i < polylines.size(); i++){
        polylines_indexs.push_back(i);
    }

    std::vector<size_t> indexs_toerase;


    for(int i = 0; i < maxSDF_polylines.size(); i++){
        if(checkif_polylineispaired(paired_indexs, i)){
            for(int j = 0; j < maxSDF_polylines[i].polyline.size(); j++){
                indexs_toerase.push_back(maxSDF_polylines[i].polyline_indexs[j]);
            }
        }
    }

    std::sort(indexs_toerase.begin(), indexs_toerase.end(), std::greater<size_t>());

    for(int i = 0; i < indexs_toerase.size(); i++){
        //std::cout << "[" << i << "]:\t" << indexs_toerase[i] << std::endl;
        polylines_copy.erase(polylines_copy.begin() + indexs_toerase[i]);
        polylines_indexs.erase(polylines_indexs.begin() + indexs_toerase[i]);
    }

    //std::cout << polylines_indexs.size() << std::endl;

    //std::cout << "123" << std::endl;
    int a = 0;

    while(!polylines_copy.empty()){
        polygon_2d tmp_newpoly;
        std::vector<size_t> polylinesindexs_toerase;
        bool flag_loop = false;
        std::vector<std::vector<int>> used_pair;
        
        for(int i = 0; i < polylines_copy.size(); i++){

            for(int j = 0; j < maxSDF_polylines.size(); j++){
                if(polylines_copy[i].back() == maxSDF_polylines[j].polyline[0].front() && checkif_polylineispaired(paired_indexs, j)){
                    used_pair.push_back(std::vector<int>{j, get_paired_polylines(paired_indexs, j)});
                    polylinesindexs_toerase.push_back(i);
                    //std::cout << i << std::endl;
                    i = std::distance(polylines_indexs.begin(), std::find(polylines_indexs.begin(), polylines_indexs.end(), maxSDF_polylines[get_paired_polylines(paired_indexs, j)].polyline_indexs.back() + 1));
                    //std::cout << i << std::endl;
                    if(i == 0) flag_loop = true;
                    break;
                }
            }

            if(flag_loop) break;

            //std::cout << "[" << i << "]:\t" << polylines_indexs[i] << std::endl;
            tmp_newpoly.outer().push_back(polylines_copy[i].front());
            tmp_newpoly.outer().push_back(polylines_copy[i].back());
            polylinesindexs_toerase.push_back(i);
        }

        //std::cout << "cutto" << std::endl;
        a++;
        //std::cout << a << std::endl;

        tmp_newpoly.outer().erase(std::unique(tmp_newpoly.outer().begin(), tmp_newpoly.outer().end()), tmp_newpoly.outer().end());
        correct(tmp_newpoly);
        child_polys.push_back(tmp_newpoly);

        std::sort(polylinesindexs_toerase.begin(), polylinesindexs_toerase.end(), std::greater<size_t>());
        for(int i = 0; i < polylinesindexs_toerase.size(); i++){
            polylines_copy.erase(polylines_copy.begin() + polylinesindexs_toerase[i]);
            polylines_indexs.erase(polylines_indexs.begin() + polylinesindexs_toerase[i]);
        }
    }

    std::cout << "\t#cut_polys: " << cut_polys.size() << std::endl;
    std::cout << "\t#child_polys: " << child_polys.size() << std::endl;

    visualize_childpolys(polygon, child_polys);

    if(N == 0){
        N++;
        adjust_length_polylines(child_polys[0]);
        calculateSDF(child_polys[0]);
    }

    if(N == 1){
        N++;
        adjust_length_polylines(child_polys[0]);
        calculateSDF(child_polys[0]);
    }

    /*if(N == 2){
        N++;
        adjust_length_polylines(cut_polys[0]);
        calculateSDF(cut_polys[0]);
    }*/

   //sliced_polys.push_back(child_polys);

    /*for(int i = 0; i < child_polys.size(); i++){
        if(child_polys[i].outer().size() < 20) return;
        adjust_length_polylines(child_polys[i]);
        calculateSDF(child_polys[i]);
    }

    for(int i = 0; i < cut_polys.size(); i++){
        if(cut_polys[i].outer().size() < 20) return;
        adjust_length_polylines(cut_polys[i]);
        calculateSDF(cut_polys[i]);
    }*/
    
}

void visualize_slicedpolys(const polygon_2d& polygon, const std::vector<std::vector<polygon_2d>>& sliced_polys){
    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        x_poly.push_back(exterior_ring(polygon)[i].x());
        y_poly.push_back(exterior_ring(polygon)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end());
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end());

    point_2d bbox_center((max_x + min_x)/2, (max_y + min_y)/2);
    double bbox_dims[] = {(max_x - min_x), (max_y - min_y)};

    const int img_scalefactor = (int)std::max(max_x - min_x, max_y - min_y)*100 > 1000 ? scale_img : 100;

    cv::Mat gray_image((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image, start, end, cv::Scalar(170), 1);
    }

    for(int j = 0; j < sliced_polys.size(); j++){
        for(int k = 0; k < sliced_polys[j].size(); k++){
            for(int i = 0; i < num_points(sliced_polys[j][k]) - 1; i++){
                double start_x = img_scalefactor * (exterior_ring(sliced_polys[j][k])[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(sliced_polys[j][k])[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                double end_x = img_scalefactor * (exterior_ring(sliced_polys[j][k])[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(sliced_polys[j][k])[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                cv::Point start(start_x, start_y);
                cv::Point end(end_x, end_y);
                cv::line(gray_image, start, end, cv::Scalar((255 - 70) * j / sliced_polys.size() + 70), 2);
            }
        }
    }

    cv::Mat heatmap;
    cv::applyColorMap(gray_image, heatmap, cv::COLORMAP_JET);
    //std::string strng_childpolys_file = "SDFchildpolys/per_" + n_per + "_gen0.jpg";
    //cv::imwrite(strng_childpolys_file, gray_image);
    cv::imshow("Sliced polys", heatmap);
}

void print_SDF(const std::vector<double>& SDF){

    std::string strng_pathfile;
    if(strng_angle_step == "0")
        strng_pathfile = "SDF_logs/SDFvalues_" + n_per + "_angle" + strng_sweep_angle + "_smoothedSDF.txt";
    else
        strng_pathfile = "SDF_logs/SDFvalues_" + n_per + "_angle" + strng_sweep_angle + "_anglestp" + strng_angle_step + "_smoothedSDF.txt";
    
    
    std::ofstream fileout(strng_pathfile, std::ios::out);
    

    for(int i = 0; i < SDF.size(); i++){
        std::cout << "SDF value for line [" << i << "]: " << SDF[i] << std::endl;
        if(flag_print_file)
            fileout << "[" << i << "]: " << SDF[i] << std::endl;
    }

    fileout.close();

}

void visualization(const polygon_2d& polygon, const std::vector<double>& SDFvalues_lines, const std::vector<double>& normalized_SDF_old){
    const double max_SDFvalue = *std::max_element(SDFvalues_lines.begin(), SDFvalues_lines.end());
    //const double max_SDFvalue = *std::max_element(normalized_SDF_old.begin(), normalized_SDF_old.end());

    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        x_poly.push_back(exterior_ring(polygon)[i].x());
        y_poly.push_back(exterior_ring(polygon)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end());
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end());

    point_2d bbox_center((max_x + min_x)/2, (max_y + min_y)/2);
    double bbox_dims[] = {(max_x - min_x), (max_y - min_y)};

    const int img_scalefactor = (int)std::max(max_x - min_x, max_y - min_y)*100 > 1000 ? scale_img : 100;

    cv::Mat gray_image((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image, start, end, cv::Scalar((SDFvalues_lines[i]*(255 - 50)/max_SDFvalue + 50)), 3);
    }

    cv::Mat heatmap;
    cv::applyColorMap(gray_image, heatmap, cv::COLORMAP_JET);

    std::string strng_pathfile;
    if(strng_angle_step == "0")
        strng_pathfile = "SDF_imgs/heatmap_" + n_per + "_angle" + strng_sweep_angle + "_smoothedSDF.jpg";
    else
        strng_pathfile = "SDF_imgs/heatmap_" + n_per + "_angle" + strng_sweep_angle + "_anglestep" + strng_angle_step + "_smoothedSDF.jpg";

    if(flag_save_img)
        cv::imwrite(strng_pathfile, heatmap);
    cv::imshow("SDF", heatmap);

}

void calculateSDF(const polygon_2d& polygon){

    // Polygon as connected lines
    std::vector<linestring_2d> polylines(num_points(polygon) - 1);
    
    for(int i = 0; i < num_points(polygon) - 1; i++){
        polylines[i].push_back(make<point_2d>(exterior_ring(polygon)[i].x(), exterior_ring(polygon)[i].y()));
        polylines[i].push_back(make<point_2d>(exterior_ring(polygon)[i+1].x(), exterior_ring(polygon)[i+1].y()));
    }

    // Normal vector angles to each line
    std::vector<double> normals_angles(polylines.size());

    for(int i = 0; i < polylines.size(); i++){
        if(polylines[i].front().x() == polylines[i].back().x())
            normals_angles[i] = 0.;
        else if(polylines[i].front().y() == polylines[i].back().y())
            normals_angles[i] = 90.;
        else
            normals_angles[i] = rad2deg(atan2(-1./((polylines[i].back().y() - polylines[i].front().y()) / (polylines[i].back().x() - polylines[i].front().x())), 1));
    }

    // SDF computing for each line  
    const double anglemean_ratio = sweep_angle == 0 ? 1 : (sweep_angle + 2.*angle_step) / sweep_angle;
    const int N_step = sweep_angle/angle_step + 1;
    std::vector<double> SDFvalues_lines(polylines.size(), 0);
    
    const double strline_step = 1;    //direction step of SDF beams

    // Normal beams as lines
    std::vector<linestring_2d> normal_beams;
    std::vector<int> index_normalbeams; //indice del segmento (polyline) da cui parte il beam
    std::vector<double> normalbeams_len;
    std::vector<double> normalbeams_angle;
    linestring_2d mid_pts;

    // Computing of the sum of weights for the mean of the SDF
    double den_mean = 0;
    if(sweep_angle == 0)
        den_mean = 1;
    else{
        for(double j = -sweep_angle/2; j <= sweep_angle/2; j += angle_step)
            den_mean += abs(abs(j) - anglemean_ratio * sweep_angle/2) / (anglemean_ratio * sweep_angle/2);
    }
    
    for(int i = 0; i < polylines.size(); i++){
    //int i = 434;{

        //std::cout << "[i]: " << i << std::endl;

        for(double j = -sweep_angle/2; j <= sweep_angle/2; j += angle_step){

            const double strlinestep_factor = 1/sqrt(1 + tan(deg2rad(normals_angles[i] + j))*tan(deg2rad(normals_angles[i] + j)));  //normalizing direction step

            linestring_2d str_line, inter_point;
            point_2d init_point((polylines[i].front().x() + polylines[i].back().x())/2, (polylines[i].front().y() + polylines[i].back().y())/2);    // middle point of each line
            
            // checking the right orientation for the direction step (must be inside the polygon)
            int extint_flag = 1;
            const double eps = 1e-5;
            if((normals_angles[i] + j) == 90){
                if(!within(make<point_2d>(init_point.x(), init_point.y() + strline_step * eps), polygon))
                    extint_flag *= -1;
                str_line.push_back(make<point_2d>(init_point.x(), init_point.y() + extint_flag * strline_step * eps));
            }
            else if((normals_angles[i] + j) == 0){
                if(!within(make<point_2d>(init_point.x() + strline_step * eps, init_point.y()), polygon))
                    extint_flag *= -1;
                str_line.push_back(make<point_2d>(init_point.x() + extint_flag * strline_step * eps, init_point.y()));
            }
            else{
                if(!within(make<point_2d>(init_point.x() + strlinestep_factor * strline_step * eps, init_point.y() + strlinestep_factor * strline_step * eps * tan(deg2rad(normals_angles[i] + j))), polygon))
                    extint_flag *= -1;
                str_line.push_back(make<point_2d>(init_point.x() + strlinestep_factor * extint_flag * strline_step * eps, init_point.y() + strlinestep_factor * extint_flag * strline_step * eps * tan(deg2rad(normals_angles[i] + j))));
            }

            // propagation of each beam (for line) until intersection
            double t = 0;
            do{
                t += extint_flag * strline_step;

                if((normals_angles[i] + j) == 90){
                    str_line.push_back(make<point_2d>(init_point.x(), init_point.y() + t));
                    intersection(str_line, polygon, inter_point);
                }
                    
                else if((normals_angles[i] + j) == 0){
                    str_line.push_back(make<point_2d>(init_point.x() + t, init_point.y()));
                    intersection(str_line, polygon, inter_point);
                }
                    
                else{
                    str_line.push_back(make<point_2d>(init_point.x() + strlinestep_factor * t, init_point.y() + strlinestep_factor * tan(deg2rad(normals_angles[i] + j)) * t));
                    intersection(str_line, polygon, inter_point);
                }
                    
            } while(inter_point.empty());


            // getting info for masks configuration
            /*if(j == 0){
                
                const double eps = 1e-4;

                int interpoint_index;
                linestring_2d tmp_interpoint;
                tmp_interpoint.push_back(make<point_2d>(inter_point.front().x() - eps * cos(deg2rad(normals_angles[i])), inter_point.front().y() - eps * sin(deg2rad(normals_angles[i]))));
                tmp_interpoint.push_back(make<point_2d>(inter_point.front().x() + eps * cos(deg2rad(normals_angles[i])), inter_point.front().y() + eps * sin(deg2rad(normals_angles[i]))));

                // getting index of the segment of the intersection point
                for(int k = 0; k < polylines.size(); k++){
                    linestring_2d interpoint_line;
                    intersection(polylines[k], tmp_interpoint, interpoint_line);
                    if(!interpoint_line.empty()){
                        interpoint_index = k;
                        break;
                    }
                }

                // checking if two beams are parallel to each other
                //if(abs(normals_angles[i] - normals_angles[interpoint_index]) < tol){
                    linestring_2d tmp;
                    tmp.push_back(init_point);
                    tmp.push_back(inter_point.front());
                    normal_beams.push_back(tmp);
                    mid_pts.push_back(make<point_2d>((init_point.x() + inter_point.front().x())/2, (init_point.y() + inter_point.front().y())/2));
                    normalbeams_len.push_back(distance(init_point, inter_point.front()));
                    normalbeams_angle.push_back(normals_angles[i] <= 0 ? normals_angles[i] + 180 : normals_angles[i]);
                    index_normalbeams.push_back(i);
                //}
            }*/


            // numerator of the SDF mean
            if(sweep_angle == 0)
                SDFvalues_lines[i] = (1./distance(init_point, inter_point.front()));
            else
                SDFvalues_lines[i] += abs(abs(j) - anglemean_ratio * sweep_angle/2) / (anglemean_ratio * sweep_angle)/2 * (1./distance(init_point, inter_point.front()));


            if(sweep_angle == 0)
                break;

        }

        // SDF values for each line
        SDFvalues_lines[i] /= den_mean;
    }

    // Calculating rects
    std::vector<rect> rects;
    linestring_2d midpts_copy = mid_pts;
    std::vector<double> len_copy = normalbeams_len;
    std::vector<double> angle_copy = normalbeams_angle;

    /*while(!midpts_copy.empty()){

        linestring_2d tmp_midpts;
        tmp_midpts.push_back(midpts_copy[0]);
        std::vector<size_t> erase_index;
        erase_index.push_back(0);
        double len_mean_num = len_copy[0];
        if(angle_copy[0] > 180 - angle_tol)
            angle_copy[0] -= 180;
        double angle_mean_num = angle_copy[0];
        double len_mean = len_copy[0];
        double angle_mean = angle_copy[0];
        
        for(int i = 1; i < midpts_copy.size(); i++){
            if(angle_copy[i] > 180 - angle_tol)
                angle_copy[i] -= 180;
            if( distance(midpts_copy[i], tmp_midpts.back()) < dist_tol && abs(len_mean - len_copy[i]) < len_tol && abs(angle_mean - angle_copy[i]) < angle_tol ){
                tmp_midpts.push_back(midpts_copy[i]);
                erase_index.push_back(i);
                len_mean_num += len_copy[i];
                len_mean = len_mean_num / tmp_midpts.size();
                angle_mean_num += angle_copy[i];
                angle_mean = angle_mean_num / tmp_midpts.size();
            }
        }

        if(angle_mean < 0)
            angle_mean += 180;

        if(tmp_midpts.size() > nbeams_tol){
            rect tmp_rect;
            double x = 0;
            double y = 0;
            std::vector<double> x_midpts, y_midpts;

            for(int i = 0; i < tmp_midpts.size(); i++){
                x += tmp_midpts[i].x();
                y += tmp_midpts[i].y();
                x_midpts.push_back(tmp_midpts[i].x());
                y_midpts.push_back(tmp_midpts[i].y());
            }
            x /= tmp_midpts.size();
            y /= tmp_midpts.size();

            tmp_rect.centr = make<point_2d>(x, y);

            const double max_x = *std::max_element(x_midpts.begin(), x_midpts.end());
            const double max_y = *std::max_element(y_midpts.begin(), y_midpts.end());
            const double min_x = *std::min_element(x_midpts.begin(), x_midpts.end());
            const double min_y = *std::min_element(y_midpts.begin(), y_midpts.end());

            double m, q;
            calculateLinearFit(tmp_midpts, m, q);

            const double len_midpts = distance(make<point_2d>(max_x, m * max_x + q), make<point_2d>(min_x, m * min_x + q));

            const double anglemean_perp = angle_mean < 90 ? angle_mean + 90 : angle_mean - 90;
            //const double anglemean_perp = rad2deg(atan2(m, 1)) <= 0 ? rad2deg(atan2(m, 1)) + 180 : rad2deg(atan2(m, 1));

            double p_mm_x = tmp_rect.centr.x() - len_mean/2 * cos(deg2rad(angle_mean)) - len_midpts/2 * cos(deg2rad(anglemean_perp));
            double p_mm_y = tmp_rect.centr.y() - len_mean/2 * sin(deg2rad(angle_mean)) - len_midpts/2 * sin(deg2rad(anglemean_perp));
            point_2d p_mm(p_mm_x, p_mm_y);
            tmp_rect.vrtxs.push_back(p_mm);

            double p_mp_x = tmp_rect.centr.x() - len_mean/2 * cos(deg2rad(angle_mean)) + len_midpts/2 * cos(deg2rad(anglemean_perp));
            double p_mp_y = tmp_rect.centr.y() - len_mean/2 * sin(deg2rad(angle_mean)) + len_midpts/2 * sin(deg2rad(anglemean_perp));
            point_2d p_mp(p_mp_x, p_mp_y);
            tmp_rect.vrtxs.push_back(p_mp);

            double p_pp_x = tmp_rect.centr.x() + len_mean/2 * cos(deg2rad(angle_mean)) + len_midpts/2 * cos(deg2rad(anglemean_perp));
            double p_pp_y = tmp_rect.centr.y() + len_mean/2 * sin(deg2rad(angle_mean)) + len_midpts/2 * sin(deg2rad(anglemean_perp));
            point_2d p_pp(p_pp_x, p_pp_y);
            tmp_rect.vrtxs.push_back(p_pp);

            double p_pm_x = tmp_rect.centr.x() + len_mean/2 * cos(deg2rad(angle_mean)) - len_midpts/2 * cos(deg2rad(anglemean_perp));
            double p_pm_y = tmp_rect.centr.y() + len_mean/2 * sin(deg2rad(angle_mean)) - len_midpts/2 * sin(deg2rad(anglemean_perp));
            point_2d p_pm(p_pm_x, p_pm_y);
            tmp_rect.vrtxs.push_back(p_pm);

            polygon_2d rect;
            const double pts[][2] = { {p_mm.x(), p_mm.y()}, {p_mp.x(), p_mp.y()}, {p_pp.x(), p_pp.y()}, {p_pm.x(), p_pm.y()}, {p_mm.x(), p_mm.y()} };
            assign_points(rect, pts);
            correct(rect);
            tmp_rect.area = area(rect);

            rects.push_back(tmp_rect);
        }

        std::sort(erase_index.begin(), erase_index.end(), std::greater<size_t>());

        for(int i = 0; i < erase_index.size(); i++){
            midpts_copy.erase(midpts_copy.begin() + erase_index[i]);
            len_copy.erase(len_copy.begin() + erase_index[i]);
            angle_copy.erase(angle_copy.begin() + erase_index[i]);
        }        

    }*/

    smoothing_SDF(SDFvalues_lines, polylines, normals_angles);

    std::vector<double> normalized_SDF = SDFvalues_lines;
    slicing_polys(polygon, normalized_SDF, polylines);
    
    
    //visualize_masks(polygon, normal_beams, index_normalbeams, polylines, normalbeams_len, normalbeams_angle, vec_interpoints, SDFvalues_lines);
    //visualize_masks_v2(polygon, normal_beams, index_normalbeams, polylines, normalbeams_len, normalbeams_angle);
    //visualize_rects(polygon, index_normalbeams, rects, mid_pts, normalbeams_len, normalbeams_angle);

    //print_SDF(SDFvalues_lines);

    //visualization(polygon, SDFvalues_lines);
    //visualization(polygon, normalized_SDF, normalized_SDF_old);
}

int main(void){

    // Polygon
    polygon_2d polygon;

    create_polygon(polygon);

    calculateSDF(polygon);

    //visualize_slicedpolys(polygon, sliced_polys);

    cv::waitKey(0);

    return 0;
}