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

const std::string n_per = "1";

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

void visualize_poly(const polygon_2d& polygon, const int& index, const std::vector<std::vector<linestring_2d>>& sweeplines){
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
        cv::line(gray_image, start, end, cv::Scalar(200), 1);
    }

    for(int i = 0; i < sweeplines.size(); i++){
        for(int j = 0; j < sweeplines[i].size(); j++){
            double start_x = img_scalefactor * (sweeplines[i][j].front().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double start_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (sweeplines[i][j].front().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            double end_x = img_scalefactor * (sweeplines[i][j].back().x() + (1.2*bbox_dims[0]/2 - bbox_center.x()));
            double end_y = (int)(1.2 * img_scalefactor * bbox_dims[1]) - img_scalefactor * (sweeplines[i][j].back().y() + (1.2*bbox_dims[1]/2 - bbox_center.y()));
            cv::Point start(start_x, start_y);
            cv::Point end(end_x, end_y);
            cv::line(gray_image, start, end, cv::Scalar(200), 1);
        }
    }

    std::string name_fig = "Rotated poly of angle " + std::to_string((int)(angle_step * index)) + "°";
    cv::imshow(name_fig, gray_image);
}

point_2d rotate_point(const point_2d& point, const double& alpha, const point_2d& cent){
    double pt_x = point.x() * std::cos(alpha) - point.y() * std::sin(alpha) + cent.x();
    double pt_y = point.x() * std::sin(alpha) + point.y() * std::cos(alpha) + cent.y();
    return make<point_2d>(pt_x, pt_y);
}

void rotate_poly(std::vector<polygon_2d>& turned_polys, const polygon_2d& polygon, const double& alpha, const point_2d& cent){
    polygon_2d tmp_poly;

    for(int i = 0; i < polygon.outer().size(); i++){
        tmp_poly.outer().push_back(rotate_point(exterior_ring(polygon)[i], alpha, cent));
    }
    correct(tmp_poly);

    turned_polys.push_back(tmp_poly);
}

void write_vertical_lines(const polygon_2d& poly, std::vector<std::vector<linestring_2d>>& sweeplines){
    std::vector<double> x_poly, y_poly;

    for(int i = 0; i < poly.outer().size(); i++){
        x_poly.push_back(exterior_ring(poly)[i].x());
        y_poly.push_back(exterior_ring(poly)[i].y());
    }

    const double max_x = *std::max_element(x_poly.begin(), x_poly.end());
    const double max_y = *std::max_element(y_poly.begin(), y_poly.end()) * 1.1;
    const double min_x = *std::min_element(x_poly.begin(), x_poly.end());
    const double min_y = *std::min_element(y_poly.begin(), y_poly.end()) * 0.9;

    //std::cout << std::endl;
    //std::cout << "min_x:\t" << min_x << std::endl;
    //std::cout << "max_x:\t" << max_x << std::endl;
    //std::cout << "min_y:\t" << min_y << std::endl;
    //std::cout << "max_y:\t" << max_y << std::endl;

    //for(int i = 1; i < (max_x - min_x)/0.1; i++){
    for(int i = 161; i < 178; i++){
        std::vector<linestring_2d> tmp_sweepline;
        linestring_2d sweepline;
        sweepline.push_back(make<point_2d>(min_x + i * sweepline_step, min_y));
        sweepline.push_back(make<point_2d>(min_x + i * sweepline_step, max_y));

        linestring_2d inter;
        intersection(poly, sweepline, inter);
        //std::cout << std::endl;
        //std::cout << "[" << i << "]:\t" << inter.size() << std::endl;
        if(inter.size() % 2 == 0){
            for(int j = 0; j < inter.size()/2; j++){
                //std::cout << '\t' << dsv(make<point_2d>((inter[2*j].x() + inter[2*j+1].x())/2, (inter[2*j].y() + inter[2*j+1].y())/2)) << std::endl;
                //std::cout << '\t' << within(make<point_2d>((inter[2*j].x() + inter[2*j+1].x())/2, (inter[2*j].y() + inter[2*j+1].y())/2), poly) << std::endl;
                if(within(make<point_2d>((inter[2*j].x() + inter[2*j+1].x())/2, (inter[2*j].y() + inter[2*j+1].y())/2), poly)){
                    linestring_2d tmp;
                    tmp.push_back(inter[2*j]);
                    tmp.push_back(inter[2*j+1]);
                    tmp_sweepline.push_back(tmp);
                }
            }
        }

        sweeplines.push_back(tmp_sweepline);
    }
}

int main(void){

    polygon_2d polygon;
    create_polygon(polygon);

    point_2d cent;
    centroid(polygon, cent);

    point_2d pt(2, 1);
    point_2d trasl(1, 1);

    std::cout << dsv(rotate_point(pt, deg2rad(90), trasl)) << std::endl;

    /*std::vector<polygon_2d> turned_polys;
    //for(int i = 0; i < angle_tot / angle_step; i++){
    int i = 17;{
        std::vector<std::vector<linestring_2d>> sweeplines;
        rotate_poly(turned_polys, polygon, deg2rad(angle_step * i), cent);
        for(int i = 0; i < exterior_ring(turned_polys[0]).size(); i++){
            std::cout << "[" << i << "]:\t" << dsv(exterior_ring(turned_polys[0])[i]) << std::endl;
        }
        //write_vertical_lines(turned_polys[0], sweeplines);
        //visualize_poly(turned_polys[0], i, sweeplines);
    }*/

    //std::cout << turned_polys.size() << std::endl;

    cv::waitKey(0);

    return 0;
}