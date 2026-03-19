#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "a3dizhu.h"

namespace py = pybind11;
using namespace a3dizhu;

PYBIND11_MODULE(a3dizhu_cpp, m) {
    m.doc() = "A3 Dizhu C++ game engine for RL training acceleration";

    py::class_<Engine>(m, "CppEngine")
        .def(py::init<>())

        .def("set_greedy_ratio", &Engine::set_greedy_ratio)
        .def("set_random_ratio", &Engine::set_random_ratio)
        .def("seed", &Engine::seed)

        .def("reset", &Engine::reset,
             "Reset game, returns first player_id")

        .def("reset_with_hands", &Engine::reset_with_hands,
             py::arg("hands"), py::arg("start_player"),
             "Reset with specific hands for parity testing")

        .def("step", &Engine::step, py::arg("action_key"),
             "Step game with action key, returns next player_id")

        .def("get_player_id", &Engine::get_player_id)
        .def("is_over", &Engine::is_over)
        .def("is_declaration_phase", &Engine::is_declaration_phase)
        .def("is_rule_agent_seat", &Engine::is_rule_agent_seat, py::arg("pid"))

        .def("encode_obs", [](const Engine& e, int player_id) {
            py::array_t<int8_t> result(STATE_DIM);
            auto buf = result.mutable_data();
            e.encode_obs(player_id, buf);
            return result;
        }, py::arg("player_id"),
           "Encode 850-dim observation for player")

        .def("get_legal_actions", [](const Engine& e) {
            auto actions = e.get_legal_actions();
            py::dict result;
            for (auto& a : actions) {
                py::array_t<int8_t> feat(ACTION_DIM);
                std::memcpy(feat.mutable_data(), a.feature, ACTION_DIM);
                result[py::str(a.key)] = feat;
            }
            return result;
        }, "Get legal actions as {key: 52d numpy array}")

        .def("get_action_feature", [](const Engine& e, const std::string& key) {
            py::array_t<int8_t> result(ACTION_DIM);
            e.get_action_feature(key, result.mutable_data());
            return result;
        }, py::arg("key"),
           "Get 52-dim action feature for a key")

        .def("get_rule_agent_action", &Engine::get_rule_agent_action,
             "Get action key for current rule agent seat")

        .def("get_payoffs", [](const Engine& e) {
            auto p = e.get_payoffs();
            return py::cast(std::vector<float>(p.begin(), p.end()));
        })

        .def("get_training_payoffs", [](const Engine& e) {
            auto p = e.get_training_payoffs();
            return py::cast(std::vector<float>(p.begin(), p.end()));
        })

        .def("get_step_rewards", [](const Engine& e, int p) {
            return py::cast(e.get_step_rewards(p));
        }, py::arg("player"))

        .def("get_aux_targets", [](const Engine& e) {
            auto targets = e.get_aux_targets();
            py::list result;
            for (int p = 0; p < NUM_PLAYERS; p++) {
                py::array_t<int64_t> arr(3);
                auto buf = arr.mutable_data();
                for (int j = 0; j < 3; j++) buf[j] = targets[p][j];
                result.append(arr);
            }
            return result;
        }        , "Get aux targets (4 players x 3 values)")
    ;

    // ─── VectorizedEngine ──────────────────────────────────────────────
    py::class_<VectorizedEngine>(m, "VectorizedEngine")
        .def(py::init<int>(), py::arg("num_envs"))
        .def("num_envs", &VectorizedEngine::num_envs)
        .def("set_greedy_ratio", &VectorizedEngine::set_greedy_ratio)
        .def("set_random_ratio", &VectorizedEngine::set_random_ratio)
        .def("seed", &VectorizedEngine::seed, py::arg("base_seed"))

        .def("reset", &VectorizedEngine::reset, py::arg("idx"))
        .def("step",  &VectorizedEngine::step,  py::arg("idx"), py::arg("action_key"))
        .def("get_player_id",       &VectorizedEngine::get_player_id,       py::arg("idx"))
        .def("is_over",             &VectorizedEngine::is_over,             py::arg("idx"))
        .def("is_rule_agent_seat",  &VectorizedEngine::is_rule_agent_seat,  py::arg("idx"), py::arg("pid"))
        .def("is_declaration_phase", &VectorizedEngine::is_declaration_phase, py::arg("idx"))
        .def("get_rule_agent_action", &VectorizedEngine::get_rule_agent_action, py::arg("idx"))

        .def("encode_obs", [](const VectorizedEngine& v, int idx, int pid) {
            py::array_t<int8_t> r(STATE_DIM);
            v.encode_obs(idx, pid, r.mutable_data());
            return r;
        }, py::arg("idx"), py::arg("player_id"))

        .def("get_action_feature", [](const VectorizedEngine& v, int idx, const std::string& k) {
            py::array_t<int8_t> r(ACTION_DIM);
            v.get_action_feature(idx, k, r.mutable_data());
            return r;
        }, py::arg("idx"), py::arg("key"))

        .def("get_training_payoffs", [](const VectorizedEngine& v, int idx) {
            auto p = v.get_training_payoffs(idx);
            return py::cast(std::vector<float>(p.begin(), p.end()));
        }, py::arg("idx"))

        .def("get_step_rewards", [](const VectorizedEngine& v, int idx, int p) {
            return py::cast(v.get_step_rewards(idx, p));
        }, py::arg("idx"), py::arg("player"))

        .def("get_aux_targets", [](const VectorizedEngine& v, int idx) {
            auto t = v.get_aux_targets(idx);
            py::list result;
            for (int p = 0; p < NUM_PLAYERS; p++) {
                py::array_t<int64_t> arr(3);
                auto buf = arr.mutable_data();
                for (int j = 0; j < 3; j++) buf[j] = t[p][j];
                result.append(arr);
            }
            return result;
        }, py::arg("idx"))

        .def("step_random", &VectorizedEngine::step_random, py::arg("idx"),
             "Pick a random legal action and step (for stagger init)")

        // One C++ call per env: advance through all rule-agent turns,
        // return (is_done, rule_obs[N,850], rule_player_ids[N])
        .def("advance_to_decision", [](VectorizedEngine& v, int idx) {
            auto [done, steps] = v.advance_to_decision(idx);
            int n = (int)steps.size();
            py::array_t<int8_t> obs({n, (int)STATE_DIM});
            auto* ptr = obs.mutable_data();
            std::vector<int> pids(n);
            for (int i = 0; i < n; i++) {
                std::memcpy(ptr + i * STATE_DIM, steps[i].obs, STATE_DIM);
                pids[i] = steps[i].player_id;
            }
            return py::make_tuple(done, obs, py::cast(pids));
        }, py::arg("idx"))

        // Batch-prepare data for GPU inference across multiple envs.
        // Returns (obs_expanded[T,850], action_flat[T,52],
        //          obs_raw[K,850], offsets[K+1], action_keys)
        .def("prepare_batch", [](const VectorizedEngine& v,
                                 const std::vector<int>& pending) {
            auto bd = v.prepare_batch(pending);
            int T = bd.total_actions, K = (int)pending.size();

            py::array_t<int8_t> obs_exp({T, (int)STATE_DIM});
            if (T > 0) std::memcpy(obs_exp.mutable_data(),
                                   bd.obs_expanded.data(), bd.obs_expanded.size());

            py::array_t<int8_t> act_flat({T, (int)ACTION_DIM});
            if (T > 0) std::memcpy(act_flat.mutable_data(),
                                   bd.action_flat.data(), bd.action_flat.size());

            py::array_t<int8_t> obs_raw({K, (int)STATE_DIM});
            if (K > 0) std::memcpy(obs_raw.mutable_data(),
                                   bd.obs_raw.data(), bd.obs_raw.size());

            py::list offsets;
            for (int o : bd.offsets) offsets.append(o);

            py::list all_keys;
            for (auto& keys : bd.action_keys) {
                py::list ek;
                for (auto& k : keys) ek.append(k);
                all_keys.append(ek);
            }
            return py::make_tuple(obs_exp, act_flat, obs_raw, offsets, all_keys);
        }, py::arg("pending_indices"))
    ;
}
