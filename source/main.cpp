// OpenGL
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glu.h>
#include <GLFW/glfw3.h>

// OpenCV
#include <opencv2/opencv.hpp>

// g2o
#define G2O_USE_VENDORED_CERES
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

// Standard
#include <chrono>
#include <cstdio>
#include <unordered_map>
#include <vector>

class Frame {
public:
    static inline int id_generator = 0;
    int id;
    cv::Mat image_grey;
    cv::Matx33d K;
    cv::Mat dist;
    std::vector<cv::KeyPoint> kps;
    cv::Mat des;
    cv::Matx33d rotation;
    cv::Matx31d translation;
public:
    Frame() = default;
    Frame(const cv::Mat& input_image_grey, const cv::Matx33d& input_K, const cv::Mat& input_dist) {
        static cv::Ptr<cv::ORB> extractor = cv::ORB::create();
        this->id = Frame::id_generator++;
        this->image_grey = input_image_grey;
        this->K = input_K;
        this->dist = input_dist;
        std::vector<cv::Point2f> corners;
        cv::goodFeaturesToTrack(image_grey, corners, 3000, 0.01, 7);
        this->kps.reserve(corners.size());
        for (const cv::Point2f& corner : corners) {
            this->kps.push_back(cv::KeyPoint(corner, 20));
        }
        extractor->compute(image_grey, this->kps, this->des);
        this->rotation = cv::Matx33d::eye();
        this->translation = cv::Matx31d::zeros();
        std::printf("Detected: %zu features\n", this->kps.size());
    }
};

class Landmark {
public:
    static inline int id_generator = 0;
    int id;
    cv::Matx31d location;
    cv::Matx31d colour;
public:
    Landmark() = default;
    Landmark(const cv::Matx31d& input_location, const cv::Matx31d& input_colour) {
        this->id = Landmark::id_generator++;
        this->location = input_location;
        this->colour = input_colour;
    }
};

class Map {
public:
    std::unordered_map<int, Frame> frames;
    std::unordered_map<int, Landmark> landmarks;
    // TODO: Ideally this should be a multi index map.
    std::unordered_map<int, std::vector<std::pair<int, int>>> observations; // key is landmark_id, pair are frame_id and kp_index
public:
    void add_frame(const Frame& frame) {
        frames[frame.id] = frame;
    }
    
    void add_landmark(const Landmark& landmark) {
        this->landmarks[landmark.id] = landmark;
    }
    
    void add_observation(const Frame& frame, const Landmark& landmark, int kp_index) {
        this->observations[landmark.id].push_back({ frame.id, kp_index });
    }

    static cv::Matx41d triangulate(const Frame& lhs_frame, const Frame& rhs_frame, const cv::Point2d& lhs_point, const cv::Point2d& rhs_point) {
        cv::Matx34d cam1 = {
            lhs_frame.rotation(0,0), lhs_frame.rotation(0,1), lhs_frame.rotation(0,2), lhs_frame.translation(0),
            lhs_frame.rotation(1,0), lhs_frame.rotation(1,1), lhs_frame.rotation(1,2), lhs_frame.translation(1),
            lhs_frame.rotation(2,0), lhs_frame.rotation(2,1), lhs_frame.rotation(2,2), lhs_frame.translation(2),
        };
        cv::Matx34d cam2 = {
            rhs_frame.rotation(0,0), rhs_frame.rotation(0,1), rhs_frame.rotation(0,2), rhs_frame.translation(0),
            rhs_frame.rotation(1,0), rhs_frame.rotation(1,1), rhs_frame.rotation(1,2), rhs_frame.translation(1),
            rhs_frame.rotation(2,0), rhs_frame.rotation(2,1), rhs_frame.rotation(2,2), rhs_frame.translation(2),
        };
        std::vector<cv::Point2d> lhs_point_normalised;
        cv::undistortPoints(std::vector<cv::Point2d>{lhs_point}, lhs_point_normalised, lhs_frame.K, lhs_frame.dist);
        std::vector<cv::Point2d> rhs_point_normalised;
        cv::undistortPoints(std::vector<cv::Point2d>{rhs_point}, rhs_point_normalised, rhs_frame.K, rhs_frame.dist);
        cv::Mat potential_landmark(4, 1, CV_64F);
        cv::triangulatePoints(cam1, cam2, lhs_point_normalised, rhs_point_normalised, potential_landmark);
        return cv::Matx41d{
            potential_landmark.at<double>(0), 
            potential_landmark.at<double>(1), 
            potential_landmark.at<double>(2),
            potential_landmark.at<double>(3)
        };
    }
    
    void optimise(int local_window , bool fix_landmarks, int rounds) {
        std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> linear_solver = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
        std::unique_ptr<g2o::BlockSolver_6_3> solver_ptr = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear_solver));
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));
        solver->setWriteDebug(false);
        g2o::SparseOptimizer opt;
        opt.setAlgorithm(solver);
        // Add frames.
        int non_fixed_poses = 0;
        int landmark_id_start = 0;
        const int local_window_below = Frame::id_generator - 1 - local_window;
        const int local_window_fixed_below = local_window_below + 1 + (fix_landmarks == false);
        for (const auto& [frame_id, frame] : this->frames) {
            if ((local_window > 0) && (frame_id < local_window_below)) {
                continue;
            }
            const g2o::Matrix3 rotation = Eigen::Map<const Eigen::Matrix3d>(frame.rotation.t().val); // Eigen is column major.
            const g2o::Vector3 translation = Eigen::Map<const Eigen::Vector3d>(frame.translation.val);
            g2o::SE3Quat se3(rotation, translation);
            g2o::VertexSE3Expmap* v_se3 = new g2o::VertexSE3Expmap();
            v_se3->setEstimate(se3);
            v_se3->setId(frame_id);
            if ((frame_id <= (fix_landmarks == false)) || ((local_window > 0) && (frame_id < local_window_fixed_below))) {
                v_se3->setFixed(true);
            }
            else {
                ++non_fixed_poses;
            }
            opt.addVertex(v_se3);
            if (frame_id > landmark_id_start) {
                landmark_id_start = frame_id + 1;
            }
        }
        // Add landmarks.
        int non_fixed_landmarks = 0;
        int non_fixed_edges = 0;
        for (const auto& [landmark_id, landmark] : this->landmarks) {
            bool in_local = false;
            for (const auto& [frame_id, kp_index] : this->observations[landmark_id]) {
                if ((local_window == 0) || (!(frame_id < local_window_below))) {
                    in_local = true;
                    break;
                }
            }
            if (!in_local) {
                continue;
            }
            const cv::Matx31d& cvpt = this->landmarks.at(landmark_id).location;
            const g2o::Vector3 pt(cvpt(0), cvpt(1), cvpt(2));
            g2o::VertexPointXYZ* v_pt = new g2o::VertexPointXYZ();
            v_pt->setId(landmark_id_start + landmark_id);
            v_pt->setEstimate(pt);
            v_pt->setMarginalized(true);
            v_pt->setFixed(fix_landmarks);
            non_fixed_landmarks += (fix_landmarks == false);
            opt.addVertex(v_pt);
            // Add edges.
            for (const auto& [frame_id, kp_index] : this->observations[landmark_id]) {
                if ((local_window > 0) && (frame_id < local_window_below)) {
                    continue;
                }
                const Frame& frame = frames.at(frame_id);
                const cv::Point2f& cvkp = frame.kps[static_cast<unsigned int>(kp_index)].pt;
                const cv::Matx33d& cvk = frame.K;
                g2o::EdgeSE3ProjectXYZ* edge = new g2o::EdgeSE3ProjectXYZ();
                edge->setVertex(0, opt.vertex(landmark_id_start + landmark_id));
                edge->setVertex(1, opt.vertex(frame_id));
                if ((frame_id > (fix_landmarks == false)) && (landmark_id >= 0)) {
                    ++non_fixed_edges;
                }
                edge->setMeasurement(g2o::Vector2(cvkp.x, cvkp.y));
                edge->setInformation(g2o::Matrix2::Identity());
                g2o::RobustKernelHuber* robust_kernel = new g2o::RobustKernelHuber();
                robust_kernel->setDelta(std::sqrt(5.991));
                edge->setRobustKernel(robust_kernel);
                edge->fx = cvk(0,0);
                edge->fy = cvk(1,1);
                edge->cx = cvk(0,2);
                edge->cy = cvk(1,2);
                opt.addEdge(edge);
            }
        }
        // Check for some invalid optimiser states.
        if (opt.vertices().empty() || opt.edges().empty()) {
            std::printf("Optimised: No edges to optimise [%zu %zu]\n", opt.vertices().size(), opt.edges().size());
            return;
        }
        if ((non_fixed_poses == 0) || ((fix_landmarks == false) && (non_fixed_landmarks == 0)) || (non_fixed_edges == 0)) {
            std::printf("Optimised: No non fixed poses [%d %d %d]\n", non_fixed_poses, non_fixed_landmarks, non_fixed_edges);
            return;
        }
        // Run the optimisation.
        opt.initializeOptimization();
        opt.computeActiveErrors();
        const double initialChi2 = opt.activeChi2();
        //opt.setVerbose(true);
        opt.optimize(rounds);
        // Apply optimised vertices to frames and landmarks.
        for (const auto& [vertex_id, vertex] : opt.vertices()) {
            if (vertex_id < landmark_id_start) {
                Frame& frame = frames.at(vertex_id);
                const g2o::VertexSE3Expmap* v_pt = static_cast<const g2o::VertexSE3Expmap*>(vertex);
                Eigen::Map<Eigen::Matrix3d>(frame.rotation.val) = v_pt->estimate().rotation().matrix().transpose(); // Eigen is column major.
                Eigen::Map<Eigen::Vector3d>(frame.translation.val) = v_pt->estimate().translation();
            }
            else {
                Landmark& landmark = this->landmarks.at(vertex_id - landmark_id_start);
                const g2o::VertexPointXYZ* v_pt = static_cast<const g2o::VertexPointXYZ*>(vertex);
                Eigen::Map<Eigen::Vector3d>(landmark.location.val) = v_pt->estimate();
            }
        }
        std::printf("Optimised: %f to %f error [%zu %zu]\n", initialChi2, opt.activeChi2(), opt.vertices().size(), opt.edges().size());
    }

    void cull() {
        const size_t landmarks_before_cull = this->landmarks.size();
        for (std::unordered_map<int, Landmark>::iterator it = this->landmarks.begin(); it != this->landmarks.end();) {
            const std::pair<int, Landmark>& landmark = *it;
            const std::vector<std::pair<int, int>>& landmark_observations = this->observations.at(landmark.first);
            bool not_seen_in_many_frames = landmark_observations.size() <= 4;
            bool not_seen_recently = landmark_observations.empty() || ((landmark_observations.back().first + 7) < Frame::id_generator);
            if (not_seen_in_many_frames && not_seen_recently) {
                this->observations.erase(landmark.first);
                it = this->landmarks.erase(it);
                continue;
            }
            float reprojection_error = 0.0f;
            for (const auto& [frame_id, kp_index] : landmark_observations) {
                const Frame& frame = this->frames.at(frame_id);
                const cv::Matx21d measured = {static_cast<double>(frame.kps[static_cast<unsigned int>(kp_index)].pt.x), static_cast<double>(frame.kps[static_cast<unsigned int>(kp_index)].pt.y)};
                const cv::Matx31d mapped = frame.K * ((frame.rotation * landmark.second.location) + frame.translation);
                const cv::Matx21d reprojected{mapped(0) / mapped(2), mapped(1) / mapped(2)};
                reprojection_error += static_cast<float>(cv::norm(measured - reprojected));
            }
            reprojection_error /= static_cast<float>(landmark_observations.size());
            if (reprojection_error > 5.991f) {
                this->observations.erase(landmark.first);
                it = this->landmarks.erase(it);
                continue;
            }
            ++it;
        }
        const size_t landmarks_after_cull = this->landmarks.size();
        std::printf("Culled: %zu points\n", landmarks_before_cull - landmarks_after_cull);
    }
};

class SLAM {
public:
    Map mapp;

public:
    void process_frame(const cv::Matx33d& K, const cv::Mat& image_grey) {
        Frame frame(image_grey, K, cv::Mat::zeros(1, 4, CV_64F));
        this->mapp.add_frame(frame);
        // Nothing to do for the first frame.
        if (frame.id == 0) {
            return;
        }
        // Get the most recent pair of frames.
        Frame& frame_current = mapp.frames.at(Frame::id_generator - 1);
        Frame& frame_previous = mapp.frames.at(Frame::id_generator - 2);
        // Helper functions for filtering matches.
        constexpr static const auto ratio_test = [](std::vector<std::vector<cv::DMatch>>& matches) {
            matches.erase(std::remove_if(matches.begin(), matches.end(), [](const std::vector<cv::DMatch>& options){
                return ((options.size() <= 1) || ((options[0].distance / options[1].distance) > 0.75f));
            }), matches.end());
        };
        constexpr static const auto symmetry_test = [](const std::vector<std::vector<cv::DMatch>> &matches1, const std::vector<std::vector<cv::DMatch>> &matches2, std::vector<std::vector<cv::DMatch>>& symMatches) {
            symMatches.clear();
            for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 = matches1.begin(); matchIterator1!= matches1.end(); ++matchIterator1) {
                for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator2 = matches2.begin(); matchIterator2!= matches2.end();++matchIterator2) {
                    if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx &&(*matchIterator2)[0].queryIdx == (*matchIterator1)[0].trainIdx) {
                        symMatches.push_back({cv::DMatch((*matchIterator1)[0].queryIdx,(*matchIterator1)[0].trainIdx,(*matchIterator1)[0].distance)});
                        break;
                    }
                }
            }
        };
        // Compute matches and filter.
        static cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
        std::vector<std::vector<cv::DMatch>> matches_cp;
        std::vector<std::vector<cv::DMatch>> matches_pc;
        matcher->knnMatch(frame_current.des, frame_previous.des, matches_cp, 2);
        matcher->knnMatch(frame_previous.des, frame_current.des, matches_pc, 2);
        ratio_test(matches_cp);
        ratio_test(matches_pc);
        std::vector<std::vector<cv::DMatch>> matches;
        symmetry_test(matches_cp, matches_pc, matches);
        // Create final arrays of good matches.
        std::vector<int> match_index_current;
        std::vector<int> match_index_previous;
        std::vector<cv::Matx21d> match_point_current;
        std::vector<cv::Matx21d> match_point_previous;
        for (const std::vector<cv::DMatch>& match : matches) {
            const cv::DMatch& m = match[0];
            match_index_current.push_back(m.queryIdx);
            match_point_current.push_back({static_cast<double>(frame_current.kps[static_cast<unsigned int>(m.queryIdx)].pt.x), static_cast<double>(frame_current.kps[static_cast<unsigned int>(m.queryIdx)].pt.y)});
            match_index_previous.push_back(m.trainIdx);
            match_point_previous.push_back({static_cast<double>(frame_previous.kps[static_cast<unsigned int>(m.trainIdx)].pt.x), static_cast<double>(frame_previous.kps[static_cast<unsigned int>(m.trainIdx)].pt.y)});
        }
        std::printf("Matched: %zu features to previous frame\n", matches.size());
        // Pose estimation of new frame.
        if (frame_current.id < 2) {
            cv::Mat E;
            cv::Mat R;
            cv::Mat t;
            int inliers = cv::recoverPose(match_point_current, match_point_previous, frame_current.K, frame_current.dist, frame_previous.K, frame_previous.dist, E, R, t, cv::RANSAC, 0.999, 1.0);
            cv::Matx33d rotation = {
                R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), 
                R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), 
                R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2),
            };
            cv::Matx31d translation = {
                t.at<double>(0),
                t.at<double>(1),
                t.at<double>(2)
            };
            // Recover pose seems to return pose 2 to pose 1 rather than pose 1 to pose 2, so invert it.
            rotation = rotation.t();
            translation = -rotation * translation;
            // Set the initial pose of the new frame.
            frame_current.rotation = rotation * frame_previous.rotation;
            frame_current.translation = frame_previous.translation + (frame_previous.rotation * translation);
            std::printf("Inliers: %d inliers in pose estimation\n", inliers);
        }
        else {
            // Set the initial pose of the new frame from the previous frame.
            frame_current.rotation = frame_previous.rotation;
            frame_current.translation = frame_previous.translation;
        }
        // Cache observations from the preious frame that are in this one.
        std::unordered_map<int, int> frame_previous_points; // key is kp_index and data is landmark_id.
        for (const auto& [landmark_id, landmark_observations] : this->mapp.observations) {
            for (const auto& [frame_id, kp_index] : landmark_observations) {
                if (frame_id == frame_previous.id) {
                    frame_previous_points[kp_index] = landmark_id;
                }
            }
        }
        // Find matches that are already in the map as landmarks.
        int observations_of_landmarks = 0;
        for (size_t i = 0; i < match_index_previous.size(); ++i) {
            auto found_point = frame_previous_points.find(match_index_previous[i]);
            if (found_point != frame_previous_points.end()) {
                // Add observations to features that match existing landmarks.
                this->mapp.add_observation(frame_current, this->mapp.landmarks[found_point->second], match_index_current[i]);
                ++observations_of_landmarks;
            }
        }
        std::printf("Matched: %d features to previous frame landmarks\n", observations_of_landmarks);
        // Optimise the pose of the new frame using the points that match the current map.
        this->mapp.optimise(1, true, 50);
        this->mapp.cull();
        // Search for matches to other frames by projection.
        int observations_of_map = 0;
        for (const auto& [landmark_id, landmark] : this->mapp.landmarks) {
            // Check landmark is infront of frame.
            const cv::Matx31d mapped = frame_current.K * ((frame_current.rotation * landmark.location) + frame_current.translation);
            const cv::Matx21d reprojected{mapped(0) / mapped(2), mapped(1) / mapped(2)};
            if ((reprojected(0) < 10) || (reprojected(0) > image_grey.cols - 11) || (reprojected(1) < 10) || (reprojected(1) > image_grey.rows - 11)) {
                continue;
            }
            // Check it has not already been matched.
            const std::vector<std::pair<int, int>>& landmark_observations = this->mapp.observations.at(landmark_id);
            if (std::find_if(landmark_observations.begin(), landmark_observations.end(), [&frame_current](const std::pair<int, int>& landmark_observation){
                return (landmark_observation.first == frame_current.id);
            }) != landmark_observations.end()) {
                continue;
            }
            // Find all detected features that are near the landmark reprojection.
            for (size_t i = 0; i < match_point_current.size(); ++i) {
                // Is it near enough?
                if (cv::norm(match_point_current[i] - reprojected) > 2) {
                    continue;
                }
                // Has it already been matched?
                auto found_point = frame_previous_points.find(match_index_previous[i]);
                if (found_point != frame_previous_points.end()) {
                    continue;
                }
                // Check similarity.
                // TODO: Only checking first descriptor.
                const cv::Mat& des_landmark = this->mapp.frames.at(landmark_observations[0].first).des.row(landmark_observations[0].second);
                const cv::Mat& des_current = frame_current.des.row(match_index_current[i]);
                if (cv::norm(des_landmark, des_current, cv::NORM_HAMMING) < 64) {
                    this->mapp.add_observation(frame_current, landmark, match_index_current[i]);
                    frame_previous_points[match_index_previous[i]] = -1;
                    ++observations_of_map;
                }
            }
        }
        std::printf("Matched: %d features to map landmarks\n", observations_of_map);
        // Optimise the pose of the new frame using reprojected points from the current map.
        this->mapp.optimise(1, true, 50);
        this->mapp.cull();
        // Add all the remaing matched features.
        int new_landmarks = 0;
        for (size_t i = 0; i < match_index_previous.size(); ++i) {
            auto found_point = frame_previous_points.find(match_index_previous[i]);
            if (found_point == frame_previous_points.end()) {
                // New landmark to be triangulated.
                cv::Matx41d point = Map::triangulate(frame_current, frame_previous, frame_current.kps[static_cast<unsigned int>(match_index_current[i])].pt, frame_previous.kps[static_cast<unsigned int>(match_index_previous[i])].pt);
                // Valid triangulation?
                if (std::abs(point(3)) < 1e-5) {
                    continue;
                }
                cv::Matx31d pt = {point(0) / point(3), point(1) / point(3), point(2) / point(3)};
                // Reproject infront of both cameras?
                const cv::Matx31d mapped_previous = frame_previous.K * ((frame_previous.rotation * pt) + frame_previous.translation);
                if (mapped_previous(2) < 0) {
                    continue;
                }
                const cv::Matx31d mapped_current = frame_current.K * ((frame_current.rotation * pt) + frame_current.translation);
                if (mapped_current(2) < 0) {
                    continue;
                }
                // Reprojection error is low?
                const cv::Matx21d reprojected_previous = {mapped_previous(0) / mapped_previous(2), mapped_previous(1) / mapped_previous(2)};
                if (cv::norm(match_point_previous[i] - reprojected_previous) > 2) {
                    continue;
                }
                const cv::Matx21d reprojected_current = {mapped_current(0) / mapped_current(2), mapped_current(1) / mapped_current(2)};
                if (cv::norm(match_point_current[i] - reprojected_current) > 2) {
                    continue;
                }
                // Add it.
                double colour = static_cast<double>(image_grey.at<unsigned char>(frame_current.kps[static_cast<unsigned int>(match_index_current[i])].pt)) / 255.0;
                Landmark landmark(pt, cv::Matx31d{colour, colour, colour});
                this->mapp.add_landmark(landmark);
                this->mapp.add_observation(frame_previous, landmark, match_index_previous[i]);
                this->mapp.add_observation(frame_current, landmark, match_index_current[i]);
                ++new_landmarks;
            }
        }
        std::printf("Created: %d new landmarks\n", new_landmarks);
        // Optimise the pose of the new frame again.
        this->mapp.optimise(1, true, 50);
        this->mapp.cull();
        // Optimise the whole map.
        this->mapp.optimise(40, false, 50);
        this->mapp.cull();
        // Print the map status and pose.
        std::printf("Map status: %zu frames, %zu landmarks\n", this->mapp.frames.size(), this->mapp.landmarks.size());
        std::printf(
            "[[% 10.8f % 10.8f % 10.8f % 10.8f]\n [% 10.8f % 10.8f % 10.8f % 10.8f]\n [% 10.8f % 10.8f % 10.8f % 10.8f]\n [% 10.8f % 10.8f % 10.8f % 10.8f]]\n",
            frame_current.rotation(0,0), frame_current.rotation(0,1), frame_current.rotation(0,2),  frame_current.translation(0),
            frame_current.rotation(1,0), frame_current.rotation(1,1), frame_current.rotation(1,2),  frame_current.translation(1),
            frame_current.rotation(2,0), frame_current.rotation(2,1), frame_current.rotation(2,2),  frame_current.translation(2),
            0.0, 0.0, 0.0, 1.0
        );
    }
};

class Display {
public:
    GLFWwindow* window = nullptr;
    GLuint image_texture;
    float image_frame[3] = { -0.9f, 0.9f, 0.4f };

public:
    ~Display() {
        glfwDestroyWindow(this->window);
        glfwTerminate();
    }

    Display(int width,int height) {
        if (!glfwInit()) {
            std::fprintf(stderr, "ERROR: GLFW failed to initialize");
            exit(1);
        }
        glfwWindowHint(GLFW_MAXIMIZED , GL_TRUE);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        this->window = glfwCreateWindow(width, height, "slam", nullptr, nullptr);
        glfwSetFramebufferSizeCallback(this->window, [](GLFWwindow*, int new_width, int new_height){ glViewport(0, 0, new_width, new_height); });
        glfwSetWindowCloseCallback(this->window, [](GLFWwindow* window_closing) { glfwSetWindowShouldClose(window_closing, GL_TRUE); });
        glfwMakeContextCurrent(this->window);
        glGenTextures(1, &this->image_texture);
    }

    cv::Mat capture() {
        glfwMakeContextCurrent(this->window);
        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);
        width = (width / 8) * 8;
        glReadBuffer(GL_FRONT);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        cv::Mat pixels(height, width, CV_8UC3);
        glReadPixels(0, 0, width, height, GL_BGR, GL_UNSIGNED_BYTE, pixels.data);
        return pixels;
    }

    void render(const Map& mapp) {
        glfwMakeContextCurrent(this->window);
        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);
        glViewport(0, 0, width, height);
        // Get the last frame processed.
        const Frame& frame_last_processed = mapp.frames.at(Frame::id_generator - 1);
        // Calculate the width and height of the inset image.
        const float ratio_image = (static_cast<float>(frame_last_processed.image_grey.cols) / static_cast<float>(frame_last_processed.image_grey.rows));
        const float ratio_screen = (static_cast<float>(width) / static_cast<float>(height));
        float image_width = 0;
        float image_height = 0;
        if (width > height) {
            image_width = this->image_frame[2];
            image_height = ratio_screen * this->image_frame[2] / ratio_image;
        } 
        else {
            image_width = ratio_image * this->image_frame[2] / ratio_screen;
            image_height = this->image_frame[2];
        }
        // OpenGL configuration.
        glDepthRange(0, 1);
        glClearColor(1.0, 1.0, 1.0, 1.0);
        glClearDepth(1.0);
        glShadeModel(GL_SMOOTH);
        glEnable(GL_DEPTH_TEST);
        glDisable(GL_LIGHTING);
        glDepthFunc(GL_LEQUAL);
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);
        glEnable(GL_BLEND);
        // Clear the screen.
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Configuration for 2D rendering.
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glColor3f(1, 1, 1);
        glDisable(GL_LIGHTING);
        glEnable(GL_TEXTURE_2D);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        // Rendering the image to the image texture.
        cv::Mat image;
        cv::drawKeypoints(frame_last_processed.image_grey, frame_last_processed.kps, image);
        glBindTexture(GL_TEXTURE_2D, this->image_texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.cols, image.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, image.data);
        glGenerateMipmap(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
        // Rendering the image texture to the screen.
        glBindTexture(GL_TEXTURE_2D, this->image_texture);
        glBegin(GL_QUADS);
        glTexCoord2f(0, 0);
        glVertex2f(this->image_frame[0], this->image_frame[1]);
        glTexCoord2f(0, 1);
        glVertex2f(this->image_frame[0], this->image_frame[1] - image_height);
        glTexCoord2f(1, 1);
        glVertex2f(this->image_frame[0] + image_width, this->image_frame[1] - image_height);
        glTexCoord2f(1, 0);
        glVertex2f(this->image_frame[0] + image_width, this->image_frame[1]);
        glEnd();
        // Configuration for 3D rendering.
        glColor3f(0, 0, 0);
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluPerspective(45, (static_cast<double>(width) / static_cast<double>(height)), 0.001, 1000.0);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        // Helper functions.
        constexpr static const auto rt_to_transform = [](const cv::Matx33f& R, const cv::Matx31f& t){
            cv::Matx44f transform = {
                R(0,0), R(0,1), R(0,2), t(0),
                R(1,0), R(1,1), R(1,2), t(1),
                R(2,0), R(2,1), R(2,2), t(2),
                0.0f,  0.0f,  0.0f, 1.0f
            };
            return transform;
        };
        constexpr static const auto opengl_coordinate_system = [](const cv::Matx44f& mat){
            cv::Matx44f swap = {
                1.0f,  0.0f,  0.0f, 0.0f,
                0.0f, -1.0f,  0.0f, 0.0f,
                0.0f,  0.0f, -1.0f, 0.0f,
                0.0f,  0.0f,  0.0f, 1.0f
            };
            return swap * mat * swap.inv();
        };
        // Render from beind the current pose.
        glMultMatrixf(
            opengl_coordinate_system(
                rt_to_transform(
                    mapp.frames.at(Frame::id_generator-1).rotation, 
                    mapp.frames.at(Frame::id_generator-1).translation
                ).t()
            ).val
        );
        glTranslatef(0, 0, -5);
        // Data for rendering camera frustra.
        const int edges[12][2] = {{0, 1}, {0, 3}, {0, 4}, {2, 1}, {2, 3}, {2, 7}, {6, 3}, {6, 4}, {6, 7}, {5, 1}, {5, 4}, {5, 7}};
        const float verticies[8][3] = {{1.0f, -1.0f, -0.9f}, {1.0f, 1.0f, -0.9f}, {-1.0f, 1.0f, -0.9f}, {-1.0f, -1.0f, -0.9f}, {0.1f, -0.1f, -0.1f}, {0.1f, 0.1f, -0.1f}, {-0.1f, -0.1f, -0.1f}, {-0.1f, 0.1f, -0.1f}};
        // Render current camera.
        const float scale_pose = 0.2f;
        glLineWidth(4);
        glBegin(GL_LINES);
        glColor3f(0.5f, 0.5f, 0.0f);
        for (int e = 0; e < 12; ++e) {
            for (int v = 0; v < 2; ++v) {
                cv::Matx41f vertex = {
                    verticies[edges[e][v]][0] * scale_pose,
                    verticies[edges[e][v]][1] * scale_pose,
                    verticies[edges[e][v]][2] * scale_pose,
                    1
                };
                cv::Matx44f pose = opengl_coordinate_system(
                    rt_to_transform(
                        mapp.frames.at(Frame::id_generator-1).rotation, 
                        mapp.frames.at(Frame::id_generator-1).translation
                    ).inv()
                );
                vertex = pose * vertex;
                glVertex3f(vertex(0), vertex(1), vertex(2));
            }
        }
        glEnd();
        // Render other cameras.
        float scale_camera = 0.2f;
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3f(0.0f, 1.0f, 0.0f);
        for (const auto& [frame_id, frame] : mapp.frames) {
            for (int e = 0; e < 12; ++e) {
                for (int v = 0; v < 2; ++v) {
                    cv::Matx41f vertex = {
                        verticies[edges[e][v]][0] * scale_camera,
                        verticies[edges[e][v]][1] * scale_camera,
                        verticies[edges[e][v]][2] * scale_camera,
                        1
                    };
                    cv::Matx44f pose = opengl_coordinate_system(
                        rt_to_transform(
                            frame.rotation, 
                            frame.translation
                        ).inv()
                    );
                    vertex = pose * vertex;
                    glVertex3f(vertex(0), vertex(1), vertex(2));
                }
            }
        }
        glEnd();
        // Render trajectory.
        glLineWidth(2);
        glBegin(GL_LINES);
        glColor3f(1.0, 0.0, 0.0);
        for (size_t index = 1; index < mapp.frames.size(); ++index) {
            for (int offset = -1; offset < 1; ++offset) {
                cv::Matx41f vertex = { 0, 0, 0, 1 };
                const Frame& frame = mapp.frames.at(static_cast<int>(index) + offset);
                cv::Matx44f pose = opengl_coordinate_system(
                    rt_to_transform(
                        frame.rotation, 
                        frame.translation
                    ).inv()
                );
                vertex = pose * vertex;
                glVertex3f(vertex(0), vertex(1), vertex(2));
            }
        }
        glEnd();
        // Render landmarks.
        glPointSize(10);
        glBegin(GL_POINTS);
        for (const auto& [landmark_id, landmark] : mapp.landmarks) {
            glColor3d(landmark.colour(0), landmark.colour(1), landmark.colour(2));
            glVertex3d(landmark.location(0), -landmark.location(1), -landmark.location(2));
        }
        glEnd();
        // Swap the buffers of the window.
        glfwSwapBuffers(this->window);
        glfwPollEvents();
        if (glfwWindowShouldClose(this->window)) {
            exit(0);
        }
    }
};

int main(int argc, char* argv[]) {
    std::printf("Launching slam...\n");
    if (argc < 3) {
        std::printf("Usage %s [video_file.ext] [focal_length]\n", argv[0]);
        exit(1);
    }
    std::printf("Loading video...\n");
    std::printf("  video_file:   %s\n", argv[1]);
    std::printf("  focal_length: %s\n", argv[2]);
    cv::VideoCapture capture(argv[1]);
    double F = std::atof(argv[2]);
    std::printf("Loading camera parameters...\n");
    int width = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int frame_count = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));
    cv::Matx33d K = {
        F, 0, static_cast<double>(width)/2.0,
        0, F, static_cast<double>(height)/2.0,
        0, 0, 1
    };
    std::printf("Loading display [%d %d]...\n", width, height);
    Display display(width, height);
    std::printf("Loading slam driver...\n");
    SLAM slam;
    std::printf("Processing frames...\n");
    int i = 0;
    while (capture.isOpened()) {
        std::printf("Starting frame %d/%d\n", ++i, frame_count);
        cv::Mat image;
        if (!capture.read(image)) {
            std::fprintf(stderr, "Failed to read frame.\n");
            break;
        }
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
        cv::resize(image, image, cv::Size(width, height));
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            slam.process_frame(K, image);
            std::chrono::duration<double> frame_duration = std::chrono::steady_clock::now() - start;
            std::printf("Processing time: %f seconds\n", frame_duration.count());
        }
        {
            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            display.render(slam.mapp);
            std::chrono::duration<double> frame_duration = std::chrono::steady_clock::now() - start;
            std::printf("Rendering time: %f seconds\n", frame_duration.count());
        }
        if  ((0)) {
            cv::Mat screen_capture = display.capture();
            cv::flip(screen_capture, screen_capture, 0);
            static cv::VideoWriter video("output.avi", cv::VideoWriter::fourcc('M','J','P','G'), 10, cv::Size(screen_capture.cols, screen_capture.rows));
            video.write(screen_capture);
        }
        std::printf("\n");
    }
}
