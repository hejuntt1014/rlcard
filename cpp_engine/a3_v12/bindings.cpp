#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "a3dizhu.h"

namespace py = pybind11;
using namespace a3dizhu;

PYBIND11_MODULE(a3dizhu_v12_cpp, m) {
    m.doc() = "A3 Dizhu engine with compact batches and 24-step history features";
    m.attr("FEATURE_VERSION") = "a3-v12-lite-v1";
    m.attr("STATE_DIM") = STATE_DIM;
    m.attr("ACTION_DIM") = ACTION_DIM;
    m.attr("OBS_STATIC_DIM") = OBS_STATIC_DIM;
    m.attr("HISTORY_LEN") = HISTORY_LEN;
    m.attr("HIST_TOKEN_DIM") = HIST_TOKEN_DIM;
    m.attr("AUX_DIM") = AUX_DIM;

    m.def("compute_afterstate", [](const std::vector<std::string>& cards,
                                  int start_val, int end_val) {
        CardSet hand = 0;
        for (const auto& key : cards) {
            int card = string_to_card(key);
            if (card < 0) throw py::value_error("Invalid card: " + key);
            if (hand & card_bit(card)) throw py::value_error("Duplicate card: " + key);
            hand |= card_bit(card);
        }
        if (start_val < 1 || start_val > 2 || end_val < 11 || end_val > 12)
            throw py::value_error("Straight limits must be 1/2 and 11/12");
        auto info = compute_afterstate(hand, start_val, end_val);
        py::dict result;
        result["remaining_count"] = info.remaining_count;
        result["singles_count"] = info.singles_count;
        result["pairs_count"] = info.pairs_count;
        result["triples_count"] = info.triples_count;
        result["fivecard_potential"] = info.fivecard_potential;
        result["min_steps"] = info.min_steps;
        result["has_s3"] = info.has_s3;
        result["has_sa"] = info.has_sa;
        result["has_rank3_single"] = info.has_rank3_single;
        result["has_rank2_single"] = info.has_rank2_single;
        result["has_straight_potential"] = info.has_straight_potential;
        result["has_flush_potential"] = info.has_flush_potential;
        result["has_sf_potential"] = info.has_sf_potential;
        result["has_threepair_or_fourone_potential"] = info.has_threepair_or_fourone_potential;
        return result;
    }, py::arg("cards"), py::arg("straight_start_val") = 1,
       py::arg("straight_end_val") = 12);

    py::class_<Engine>(m, "CppEngine", py::module_local())
        .def(py::init<>())

        .def("set_greedy_ratio", &Engine::set_greedy_ratio)
        .def("set_random_ratio", &Engine::set_random_ratio)
        .def("seed", &Engine::seed)

        .def("reset", &Engine::reset,
             "Reset game, returns first player_id")

        .def("reset_with_hands", &Engine::reset_with_hands,
             py::arg("hands"), py::arg("start_player"),
             "Reset with specific hands for parity testing")

        .def("set_rules", &Engine::set_rules,
             py::arg("declare_require_both_spades"),
             py::arg("straight_start_val"),
             py::arg("straight_end_val"))

        .def("step", &Engine::step, py::arg("action_key"),
             "Step game with action key, returns next player_id")

        .def("get_player_id", &Engine::get_player_id)
        .def("is_over", &Engine::is_over)
        .def("is_declaration_phase", &Engine::is_declaration_phase)
        .def("get_mode", &Engine::get_mode)
        .def("is_rule_agent_seat", &Engine::is_rule_agent_seat, py::arg("pid"))

        .def("encode_obs", [](const Engine& e, int player_id) {
            py::array_t<int8_t> result(STATE_DIM);
            auto buf = result.mutable_data();
            e.encode_obs(player_id, buf);
            return result;
        }, py::arg("player_id"),
           "Encode V11 observation (obs_static 556D + hist_tokens 24x88D = 2668D)")

        .def("get_legal_actions", [](const Engine& e) {
            auto actions = e.get_legal_actions();
            py::dict result;
            for (auto& a : actions) {
                py::array_t<int8_t> feat(ACTION_DIM);
                std::memcpy(feat.mutable_data(), a.feature, ACTION_DIM);
                result[py::str(a.key)] = feat;
            }
            return result;
        }, "Get legal actions as {key: 111d numpy array}")

        .def("get_action_feature", [](const Engine& e, const std::string& key) {
            py::array_t<int8_t> result(ACTION_DIM);
            e.get_action_feature(key, result.mutable_data());
            return result;
        }, py::arg("key"),
           "Get 111-dim V11 action feature for a key")

        .def("get_rule_agent_action", &Engine::get_rule_agent_action)

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
                // Pack V11 aux: relation(3) + s3_owner(1) + sa_owner(1) = 5 int64
                py::array_t<int64_t> arr(5);
                auto buf = arr.mutable_data();
                for (int j = 0; j < 3; j++) buf[j] = targets[p].relation[j];
                buf[3] = targets[p].s3_owner;
                buf[4] = targets[p].sa_owner;
                result.append(arr);
            }
            return result;
        }, "Get V11 aux targets (4 players x 5 values: relation[3] + s3_owner + sa_owner)")
    ;

    // ─── VectorizedEngine ──────────────────────────────────────────────
    py::class_<VectorizedEngine>(m, "VectorizedEngine", py::module_local())
        .def(py::init<int>(), py::arg("num_envs"))
        .def("num_envs", &VectorizedEngine::num_envs)
        .def("set_greedy_ratio", &VectorizedEngine::set_greedy_ratio)
        .def("set_random_ratio", &VectorizedEngine::set_random_ratio)
        .def("seed", &VectorizedEngine::seed, py::arg("base_seed"))

        .def("reset", &VectorizedEngine::reset, py::arg("idx"))
        .def("set_rules", &VectorizedEngine::set_rules, py::arg("idx"),
             py::arg("declare_require_both_spades"), py::arg("straight_start_val"),
             py::arg("straight_end_val"))
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

        .def("get_payoffs", [](const VectorizedEngine& v, int idx) {
            auto p = v.get_payoffs(idx);
            return py::cast(std::vector<float>(p.begin(), p.end()));
        }, py::arg("idx"))

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
                py::array_t<int64_t> arr(5);
                auto buf = arr.mutable_data();
                for (int j = 0; j < 3; j++) buf[j] = t[p].relation[j];
                buf[3] = t[p].s3_owner;
                buf[4] = t[p].sa_owner;
                result.append(arr);
            }
            return result;
        }, py::arg("idx"))

        .def("get_aux_targets_for_player", [](const VectorizedEngine& v, int idx, int pid) {
            auto t = v.get_aux_targets_for_player(idx, pid);
            py::array_t<int64_t> result(5);
            auto out = result.mutable_data();
            for (int j = 0; j < 3; j++) out[j] = t.relation[j];
            out[3] = t.s3_owner;
            out[4] = t.sa_owner;
            return result;
        }, py::arg("idx"), py::arg("player_id"))

        .def("step_random", &VectorizedEngine::step_random, py::arg("idx"))

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

        .def("advance_to_decision_with_data", [](VectorizedEngine& v, int idx) {
            auto [done, steps] = v.advance_to_decision(idx);
            int n = (int)steps.size();
            py::array_t<int8_t> obs({n, (int)STATE_DIM});
            py::array_t<int8_t> actions({n, (int)ACTION_DIM});
            py::array_t<int64_t> auxiliary({n, 5});
            std::vector<int> pids(n);
            for (int i = 0; i < n; i++) {
                std::memcpy(obs.mutable_data() + i * STATE_DIM, steps[i].obs, STATE_DIM);
                std::memcpy(actions.mutable_data() + i * ACTION_DIM, steps[i].action, ACTION_DIM);
                auto out = auxiliary.mutable_data() + i * 5;
                for (int j = 0; j < 3; j++) out[j] = steps[i].auxiliary.relation[j];
                out[3] = steps[i].auxiliary.s3_owner;
                out[4] = steps[i].auxiliary.sa_owner;
                pids[i] = steps[i].player_id;
            }
            return py::make_tuple(done, obs, py::cast(pids), actions, auxiliary);
        }, py::arg("idx"))

        .def("prepare_batch", [](const VectorizedEngine& v,
                                 const std::vector<int>& pending) {
            auto bd = v.prepare_batch(pending);
            int T = bd.total_actions, K = (int)pending.size();

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
            return py::make_tuple(obs_raw, act_flat, offsets, all_keys);
        }, py::arg("pending_indices"))
    ;
}
