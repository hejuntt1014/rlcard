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
        }, "Get aux targets (4 players x 3 values)")
    ;
}
