// Compile with     g++ -o angled_masks angled_masks.cpp `pkg-config --cflags --libs opencv`

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

using namespace boost::geometry;

typedef model::d2::point_xy<double> point_2d;
typedef model::linestring<point_2d> linestring_2d;
typedef model::polygon<point_2d> polygon_2d;
typedef model::box<point_2d> box_2d;

const std::string n_per = "42bis";

const bool flag_save_summask = false;

const double angle_tot = 180;       // [degrees]
const double angle_step = 10;       // [degrees]

const double sweepline_step = 0.1;  // [m]

const int scale_img = 35;

inline double rad2deg(double alpha){
    return alpha*180/M_PI;
}

inline double deg2rad(double alpha){
    return alpha*M_PI/180;
}

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

double segmentTime(const double l, const double a, const double d, const double vmax) {
    // Time from 0 to Vmax
    double t_acc = vmax / a;
    
    // Time from Vmax to 0
    double t_dec = vmax / d;
    
    // Len needed to acc from 0 to Vmax
    double l_acc = (vmax * vmax) / (2 * a);
    
    // Len needed to dec from Vmax to 0
    double l_dec = (vmax * vmax) / (2 * d);
    
    // Tot dist for triangular speed profile
    double l_max = l_acc + l_dec;
    
    if (l_max >= l) {
        return sqrt((2 * l) / a) + sqrt((2 * l) / d);
    } else {
        double t_costante = (l - l_max) / vmax;        
        return t_acc + t_dec + t_costante;
    }
}

void visualize_rotated_masks(const polygon_2d& polygon, const int& index, const std::vector<std::vector<linestring_2d>>& sweeplines){
    std::vector<double> x_poly, y_poly;

    int cont = 0;
    for(int i = 0; i < sweeplines.size(); i++){
        for(int j = 0; j < sweeplines[i].size(); j++){
            cont++;
        }
    }

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

    std::vector<std::vector<double>> sweep_distances;
    for(int i = 0; i < sweeplines.size(); i++){
        std::vector<double> tmp_dist;
        for(int j = 0; j < sweeplines[i].size(); j++){
            tmp_dist.push_back(distance(sweeplines[i][j].front(), sweeplines[i][j].back()));
        }
        sweep_distances.push_back(tmp_dist);
    }

    std::vector<double> max_distances;
    for(int i = 0; i < sweep_distances.size(); i++){
        max_distances.push_back(*std::max_element(sweep_distances[i].begin(), sweep_distances[i].end()));
    }

    double max_dist = *std::max_element(max_distances.begin(), max_distances.end());

    for(int i = 0; i < sweeplines.size(); i++){
        for(int j = 0; j < sweeplines[i].size(); j++){
            double start_x = img_scalefactor * (sweeplines[i][j].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (sweeplines[i][j].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            double end_x = img_scalefactor * (sweeplines[i][j].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (sweeplines[i][j].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            cv::Point start(start_x, start_y);
            cv::Point end(end_x, end_y);
            cv::line(gray_image, start, end, cv::Scalar((255 - 70) * distance(sweeplines[i][j].front(), sweeplines[i][j].back()) / max_dist + 70), 2);
        }
    }

    

    cv::Mat heatmap;
    cv::applyColorMap(gray_image, heatmap, cv::COLORMAP_JET);
    cv::putText(heatmap, std::to_string(cont), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(170, 170, 170), 3, 8, false);
    std::string name_fig = "Rotated poly of angle " + std::to_string((int)(angle_step * index)) + "°";
    cv::imshow(name_fig, heatmap);
}

void visualize_sum_mask(const polygon_2d& polygon, const std::vector<std::vector<std::vector<linestring_2d>>>& all_sweeplines, const std::vector<double>& distances){
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

    cv::Mat gray_image_min((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));
    cv::Mat gray_image_max((int)(1.2 * img_scalefactor * bbox_dims[1]), (int)(1.2 * img_scalefactor * bbox_dims[0]), CV_8UC1, cv::Scalar(0));

    for(int i = 0; i < num_points(polygon) - 1; i++){
        double start_x = img_scalefactor * (exterior_ring(polygon)[i].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        double end_x = img_scalefactor * (exterior_ring(polygon)[i+1].x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (exterior_ring(polygon)[i+1].y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
        cv::Point start(start_x, start_y);
        cv::Point end(end_x, end_y);
        cv::line(gray_image_min, start, end, cv::Scalar(170), 1);
        cv::line(gray_image_max, start, end, cv::Scalar(170), 1);
    }


    std::vector<double> dist_sorted_min = distances;
    std::vector<double> dist_sorted_max = distances;
    std::sort(dist_sorted_min.begin(), dist_sorted_min.end(), std::greater<double>());
    std::sort(dist_sorted_max.begin(), dist_sorted_max.end());
    //const double max_dist = dist_sorted[dist_sorted.size() - 1];

    for(int i = 0; i < dist_sorted_min.size(); i++){
        std::cout << i << std::endl;
        for(int j = 0; j < all_sweeplines.size(); j++){
            for(int k = 0; k < all_sweeplines[j].size(); k++){
                for(int l = 0; l < all_sweeplines[j][k].size(); l++){
                    if(dist_sorted_min[i] == distance(all_sweeplines[j][k][l].front(), all_sweeplines[j][k][l].back())){
                        double start_x = img_scalefactor * (all_sweeplines[j][k][l].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (all_sweeplines[j][k][l].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                        double end_x = img_scalefactor * (all_sweeplines[j][k][l].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (all_sweeplines[j][k][l].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                        cv::Point start(start_x, start_y);
                        cv::Point end(end_x, end_y);
                        cv::line(gray_image_min, start, end, cv::Scalar((255) * (j + 1) * angle_step / angle_tot), 2);
                    }
                    if(dist_sorted_max[i] == distance(all_sweeplines[j][k][l].front(), all_sweeplines[j][k][l].back())){
                        double start_x = img_scalefactor * (all_sweeplines[j][k][l].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                        double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (all_sweeplines[j][k][l].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                        double end_x = img_scalefactor * (all_sweeplines[j][k][l].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                        double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (all_sweeplines[j][k][l].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                        cv::Point start(start_x, start_y);
                        cv::Point end(end_x, end_y);
                        cv::line(gray_image_max, start, end, cv::Scalar((255) * (j + 1) * angle_step / angle_tot), 2);
                    }
                }
            }
        }
    }
    
    /*for(int i = 0; i < dist_sorted.size(); i++){
        for(int j = 0; j < distances.size(); j++){
            if(dist_sorted[i] == distances[j]){
                double start_x = img_scalefactor * (all_sweeplines[j].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (all_sweeplines[j].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                double end_x = img_scalefactor * (all_sweeplines[j].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
                double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (all_sweeplines[j].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
                cv::Point start(start_x, start_y);
                cv::Point end(end_x, end_y);
                cv::line(gray_image, start, end, cv::Scalar((255) * distances[j] / max_dist), 2);
            }
        }
    }*/

    cv::Mat heatmap_min, heatmap_max;
    cv::applyColorMap(gray_image_min, heatmap_min, cv::COLORMAP_JET);
    cv::applyColorMap(gray_image_max, heatmap_max, cv::COLORMAP_JET);
    std::string strng_img = "immagini/sum_masks/summask_" + n_per + ".jpg";
    //if(flag_save_summask)
        //cv::imwrite(strng_img, heatmap);
    cv::imshow("Sum mask min", heatmap_min);
    cv::imshow("Sum mask max", heatmap_max);
}

point_2d rotate_point(const point_2d& point, const double& alpha, const point_2d& cent){
    double pt_x = (point.x() - cent.x()) * std::cos(alpha) - (point.y() - cent.y()) * std::sin(alpha) + cent.x();
    double pt_y = (point.x() - cent.x()) * std::sin(alpha) + (point.y() - cent.y()) * std::cos(alpha) + cent.y();
    return make<point_2d>(pt_x, pt_y);
}

void rotate_poly(polygon_2d& tmp_poly, const polygon_2d& polygon, const double& alpha, const point_2d& cent){
    for(int i = 0; i < polygon.outer().size(); i++){
        tmp_poly.outer().push_back(rotate_point(exterior_ring(polygon)[i], alpha, cent));
    }
    correct(tmp_poly);

}

void order_ydec(linestring_2d& ls){
    std::sort(ls.begin(), ls.end(), [](const point_2d& a, const point_2d& b){
        return get<1>(a) > get<1>(b);
    });
}

void write_vertical_lines(const polygon_2d& poly, std::vector<std::vector<linestring_2d>>& sweeplines){
    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < poly.outer().size(); i++){
        x_poly.push_back(exterior_ring(poly)[i].x());
        y_poly.push_back(exterior_ring(poly)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end()) * 1.2;
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end()) * 0.8;

    //std::cout << std::endl;
    //std::cout << "min_x:\t" << min_x << std::endl;
    //std::cout << "max_x:\t" << max_x << std::endl;
    //std::cout << "min_y:\t" << min_y << std::endl;
    //std::cout << "max_y:\t" << max_y << std::endl;

    for(int i = 1; i < (max_x - min_x)/0.1; i++){
    //for(int i = 120; i < 128; i++){
        std::vector<linestring_2d> tmp_sweepline;

        linestring_2d sweepline;
        sweepline.push_back(make<point_2d>(min_x + i * sweepline_step, min_y));
        sweepline.push_back(make<point_2d>(min_x + i * sweepline_step, max_y));

        linestring_2d inter;
        intersection(poly, sweepline, inter);
        order_ydec(inter);
        //if(inter.size() % 2 == 0){
            for(int j = 0; j < inter.size(); j+=2){
                linestring_2d tmp;
                tmp.push_back(inter[j]);
                tmp.push_back(inter[j+1]);
                tmp_sweepline.push_back(tmp);
            }
        //}

        sweeplines.push_back(tmp_sweepline);
    }
}

void rotate_vertical_lines(std::vector<std::vector<linestring_2d>>& sweeplines, const double& alpha, const point_2d& cent){

    std::vector<std::vector<linestring_2d>> tmp_sweeplines;

    for(int i = 0; i < sweeplines.size(); i++){
        std::vector<linestring_2d> tmp_sweepline;
        for(int j = 0; j < sweeplines[i].size(); j++){
            linestring_2d tmp;
            tmp.push_back(rotate_point(sweeplines[i][j].front(), alpha, cent));
            tmp.push_back(rotate_point(sweeplines[i][j].back(), alpha, cent));
            tmp_sweepline.push_back(tmp);
        }
        tmp_sweeplines.push_back(tmp_sweepline);
    }

    sweeplines = tmp_sweeplines;
}

int main(void){

    polygon_2d polygon;
    create_polygon(polygon);

    point_2d cent;
    centroid(polygon, cent);

    std::vector<std::vector<std::vector<linestring_2d>>> all_sweeplines;
    std::vector<double> distances;
    //std::vector<linestring_2d> all_sweeplines;
    for(int i = 0; i < angle_tot / angle_step; i++){
    //int i = 16;{
        polygon_2d tmp_poly;
        std::vector<std::vector<linestring_2d>> sweeplines;
        rotate_poly(tmp_poly, polygon, deg2rad(-angle_step * i), cent);
        write_vertical_lines(tmp_poly, sweeplines);
        rotate_vertical_lines(sweeplines, deg2rad(angle_step * i), cent);
        visualize_rotated_masks(polygon, i, sweeplines);
        for(int j = 0; j < sweeplines.size(); j++){
            for(int k = 0; k < sweeplines[j].size(); k++){
                distances.push_back(distance(sweeplines[j][k].front(), sweeplines[j][k].back()));
            }
        }
        all_sweeplines.push_back(sweeplines);
    }

    //std::cout << distances.size() << std::endl;

    //visualize_sum_mask(polygon, all_sweeplines, distances);

    cv::waitKey(0);

    return 0;
}