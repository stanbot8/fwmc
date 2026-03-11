#ifndef FWMC_VIEWER_BODY_VIEWPORT_H_
#define FWMC_VIEWER_BODY_VIEWPORT_H_

// MuJoCo fly body viewport for the FWMC brain viewer.
//
// Embeds a NeuroMechFly body simulation directly in the viewer's GL context.
// The brain's motor output drives the fly's CPG, which drives MuJoCo actuators.
// Rendering uses mjr_render into a caller-specified viewport rect.
//
// This is an optional component: if FWMC_BODY_SIM is not defined, the struct
// is a no-op stub.

#ifdef FWMC_BODY_SIM

#include <mujoco/mujoco.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>

// NMF CPG (from nmfly package). Only needs nmf_cpg.h + nmf_walk_data.h.
#include "nmf_cpg.h"

namespace fwmc {

struct BodyViewport {
    mjModel*   mj_model = nullptr;
    mjData*    mj_data  = nullptr;
    mjvCamera  cam  = {};
    mjvOption  opt  = {};
    mjvScene   scn  = {};
    mjrContext con  = {};

    nmfly::NmfCpg cpg;
    bool ready      = false;
    int  step_count = 0;
    float dt        = 0.0001f;  // match model's authored timestep
    int   substeps  = 10;

    // Find the NMF model XML in several search locations.
    static std::string FindModel() {
        namespace fs = std::filesystem;
        const char* candidates[] = {
            "data/nmf_complete.xml",
            "../data/nmf_complete.xml",
            "../../flygame/cpp/data/nmf_complete.xml",
            "../flygame/cpp/data/nmf_complete.xml",
        };
        for (auto& c : candidates) {
            if (fs::exists(c))
                return fs::canonical(c).string();
        }
        return "";
    }

    bool Init(const std::string& model_path = "") {
        std::string path = model_path.empty() ? FindModel() : model_path;
        if (path.empty()) {
            printf("[body] NMF model not found, body sim disabled\n");
            return false;
        }

        char error[1024] = {};
        mj_model = mj_loadXML(path.c_str(), nullptr, error, sizeof(error));
        if (!mj_model) {
            fprintf(stderr, "[body] MuJoCo load error: %s\n", error);
            return false;
        }

        mj_model->opt.timestep = dt;
        mj_data = mj_makeData(mj_model);

        // Visualization objects (in the existing GL context).
        mjv_defaultCamera(&cam);
        mjv_defaultOption(&opt);
        mjv_defaultScene(&scn);
        mjr_defaultContext(&con);

        mjv_makeScene(mj_model, &scn, 5000);
        mjr_makeContext(mj_model, &con, mjFONTSCALE_100);

        // Camera: free orbit, looking at model center.
        cam.type = mjCAMERA_FREE;
        cam.lookat[0] = mj_model->stat.center[0];
        cam.lookat[1] = mj_model->stat.center[1];
        cam.lookat[2] = mj_model->stat.center[2];
        cam.distance  = mj_model->stat.extent * 0.6;
        cam.elevation = -20.0;
        cam.azimuth   = 135.0;

        printf("[body] MuJoCo ready (%d actuators, %d DOF)\n",
               mj_model->nu, mj_model->nv);

        // NMF-specific init: tripod pose + warmup.
        if (mj_model->nu == 48) {
            cpg.Init(16.0, 0.0);

            // Set tripod standing pose.
            double walk_ctrl[48];
            cpg.GetCtrl(walk_ctrl);
            for (int i = 0; i < 42; ++i) {
                int jnt_id = mj_model->actuator_trnid[2 * i];
                if (jnt_id >= 0 && jnt_id < mj_model->njnt) {
                    int qadr = mj_model->jnt_qposadr[jnt_id];
                    mj_data->qpos[qadr] = walk_ctrl[i];
                }
                mj_data->ctrl[i] = walk_ctrl[i];
            }
            for (int i = 42; i < 48; ++i)
                mj_data->ctrl[i] = 1.0;  // adhesion on (flygym uses 0/1)
            for (int i = 0; i < mj_model->nv; ++i)
                mj_data->qvel[i] = 0.0;

            mj_forward(mj_model, mj_data);

            // Warmup: hold tripod pose via actuators while contacts settle.
            for (int s = 0; s < 500; ++s)
                mj_step(mj_model, mj_data);

            printf("[body] NMF settled (z=%.4f, ncon=%d)\n",
                   mj_data->qpos[2], mj_data->ncon);
        }

        ready = true;
        return true;
    }

    // Step physics. Drive is set externally via cpg.SetDrive().
    void Step() {
        if (!ready || mj_model->nu != 48) return;

        // Integrate CPG per physics substep.
        double sub_dt = static_cast<double>(dt);
        bool stable = true;
        for (int s = 0; s < substeps; ++s) {
            cpg.Step(sub_dt);
            double ctrl[48];
            cpg.GetCtrl(ctrl);
            for (int i = 0; i < 48; ++i)
                mj_data->ctrl[i] = ctrl[i];
            mj_step(mj_model, mj_data);

            // Stability check.
            for (int j = 0; j < mj_model->nv; ++j) {
                if (!std::isfinite(mj_data->qacc[j])) {
                    stable = false;
                    break;
                }
            }
            if (!stable) break;
        }

        if (!stable) {
            // Reset to tripod pose.
            mj_resetData(mj_model, mj_data);
            cpg.Init(16.0, 0.0);
            double reset_ctrl[48];
            cpg.GetCtrl(reset_ctrl);
            for (int i = 0; i < 42; ++i) {
                int jnt_id = mj_model->actuator_trnid[2 * i];
                if (jnt_id >= 0 && jnt_id < mj_model->njnt) {
                    int qadr = mj_model->jnt_qposadr[jnt_id];
                    mj_data->qpos[qadr] = reset_ctrl[i];
                }
                mj_data->ctrl[i] = reset_ctrl[i];
            }
            for (int i = 42; i < 48; ++i)
                mj_data->ctrl[i] = 1.0;
            for (int i = 0; i < mj_model->nv; ++i)
                mj_data->qvel[i] = 0.0;
            mj_forward(mj_model, mj_data);
            // Warmup after reset.
            for (int s = 0; s < 500; ++s)
                mj_step(mj_model, mj_data);
            cpg.SetDrive(1.0, 0.0);  // resume walking
            printf("[body] Physics reset (instability)\n");
            return;
        }

        step_count += substeps;
    }

    // Smoothed camera target (filters out physics micro-vibrations).
    double smooth_lookat[3] = {0, 0, 0};
    bool lookat_init = false;

    // Render to the given viewport rect in the current GL context.
    void Render(mjrRect viewport) {
        if (!ready) return;

        // Smooth follow camera: EMA on fly position to remove jitter.
        constexpr double kCamAlpha = 0.05;  // lower = smoother
        if (!lookat_init) {
            smooth_lookat[0] = mj_data->qpos[0];
            smooth_lookat[1] = mj_data->qpos[1];
            smooth_lookat[2] = mj_data->qpos[2];
            lookat_init = true;
        } else {
            for (int i = 0; i < 3; ++i)
                smooth_lookat[i] += kCamAlpha * (mj_data->qpos[i] - smooth_lookat[i]);
        }
        cam.lookat[0] = smooth_lookat[0];
        cam.lookat[1] = smooth_lookat[1];
        cam.lookat[2] = smooth_lookat[2];

        mjr_setBuffer(mjFB_WINDOW, &con);
        mjv_updateScene(mj_model, mj_data, &opt, nullptr, &cam,
                        mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);
    }

    // Camera interaction (call from mouse callbacks).
    void RotateCamera(double dx, double dy, int width, int height) {
        if (!ready) return;
        mjv_moveCamera(mj_model, mjMOUSE_ROTATE_V,
                        dx / width, dy / height, &scn, &cam);
    }

    void TranslateCamera(double dx, double dy, int width, int height) {
        if (!ready) return;
        mjv_moveCamera(mj_model, mjMOUSE_MOVE_V,
                        dx / width, dy / height, &scn, &cam);
    }

    void ZoomCamera(double dy, int /*height*/) {
        if (!ready) return;
        mjv_moveCamera(mj_model, mjMOUSE_ZOOM,
                        0, -0.05 * dy, &scn, &cam);
    }

    float SimTime() const { return step_count * dt; }

    // Proprioceptive feedback: yaw rate of the fly body (rad/s).
    // Positive = turning left (counterclockwise when viewed from above).
    // This is what the halteres sense in a real fly.
    float YawRate() const {
        if (!ready || !mj_data) return 0.0f;
        // The free joint's angular velocity is in qvel[3..5] (wx, wy, wz).
        // Z-axis is up, so qvel[5] is the yaw rate.
        return static_cast<float>(mj_data->qvel[5]);
    }

    void Shutdown() {
        if (!ready) return;
        mjr_freeContext(&con);
        mjv_freeScene(&scn);
        if (mj_data)  { mj_deleteData(mj_data);   mj_data = nullptr; }
        if (mj_model) { mj_deleteModel(mj_model); mj_model = nullptr; }
        ready = false;
    }

    ~BodyViewport() { Shutdown(); }

    // Non-copyable.
    BodyViewport() = default;
    BodyViewport(const BodyViewport&) = delete;
    BodyViewport& operator=(const BodyViewport&) = delete;
};

}  // namespace fwmc

#else  // !FWMC_BODY_SIM

// Stub when MuJoCo is not available.
namespace fwmc {
struct BodyViewport {
    bool ready = false;
    bool Init(const std::string& = "") { return false; }
    void Step() {}
    void Render(int, int, int, int) {}
    void RotateCamera(double, double, int, int) {}
    void TranslateCamera(double, double, int, int) {}
    void ZoomCamera(double, int) {}
    float SimTime() const { return 0.0f; }
    void Shutdown() {}
};
}  // namespace fwmc

#endif  // FWMC_BODY_SIM

#endif  // FWMC_VIEWER_BODY_VIEWPORT_H_
