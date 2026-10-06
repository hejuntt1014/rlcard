#pragma once

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include "a3dizhu.h"

namespace a3dizhu_v12 {
namespace py = pybind11;

// Decision-time data stays in contiguous native columns until episode completion.
// Finished arrays take ownership of their storage, independently of this buffer.
class RolloutBuffer {
    using ByteArray = py::array_t<int8_t, py::array::c_style>;
    using LabelArray = py::array_t<int64_t, py::array::c_style>;
    struct Columns {
        std::vector<int8_t> state, action;
        std::vector<int64_t> auxiliary;
        size_t size() const { return auxiliary.size() / 5; }
        void append(const int8_t* obs, const int8_t* act, const int64_t* aux) {
            // A typical A3 role has tens of decisions. Reserve once instead of
            // repeatedly copying the growing observation column at 1/2/4/8/16.
            if (state.capacity() == 0) {
                state.reserve(32 * STATE_DIM);
                action.reserve(32 * ACTION_DIM);
                auxiliary.reserve(32 * 5);
            }
            state.insert(state.end(), obs, obs + STATE_DIM);
            action.insert(action.end(), act, act + ACTION_DIM);
            auxiliary.insert(auxiliary.end(), aux, aux + 5);
        }
    };
    std::vector<std::array<Columns, NUM_PLAYERS>> episodes_;
    std::vector<size_t> counts_;
    size_t max_steps_;

    void check_index(int idx) const {
        if (idx < 0 || idx >= (int)episodes_.size())
            throw py::index_error("Environment index out of range");
    }
    void check_capacity(int idx, size_t additional) const {
        if (additional > max_steps_ || counts_[idx] > max_steps_ - additional)
            throw std::runtime_error("Episode exceeded max_episode_steps");
    }
    static void check_arrays(const ByteArray& obs, const ByteArray& actions,
                             const LabelArray& auxiliary, size_t rows) {
        if (obs.ndim() != 2 || obs.shape(0) != (py::ssize_t)rows || obs.shape(1) != STATE_DIM
            || actions.ndim() != 2 || actions.shape(1) != ACTION_DIM
            || auxiliary.ndim() != 2 || auxiliary.shape(0) != (py::ssize_t)rows || auxiliary.shape(1) != 5)
            throw py::value_error("Rollout arrays have incompatible shapes");
    }

public:
    explicit RolloutBuffer(int num_envs, int max_episode_steps = 10000) {
        if (num_envs < 1 || max_episode_steps < 1)
            throw py::value_error("Environment count and max_episode_steps must be positive");
        episodes_.resize(num_envs);
        counts_.resize(num_envs, 0);
        max_steps_ = max_episode_steps;
    }

    size_t size(int idx) const { check_index(idx); return counts_[idx]; }

    void clear(int idx) {
        check_index(idx);
        episodes_[idx] = {};
        counts_[idx] = 0;
    }

    void record_choices(const std::vector<int>& indices, const std::vector<int>& choices,
                        const ByteArray& obs, const ByteArray& actions,
                        const std::vector<int>& offsets, const std::vector<int>& players,
                        const LabelArray& auxiliary) {
        const size_t rows = indices.size();
        check_arrays(obs, actions, auxiliary, rows);
        if (choices.size() != rows || players.size() != rows || offsets.size() != rows + 1
            || offsets.front() != 0 || offsets.back() != actions.shape(0))
            throw py::value_error("One choice, player and candidate group is required per environment");
        std::vector<bool> seen(episodes_.size(), false);
        for (size_t row = 0; row < rows; ++row) {
            const int idx = indices[row], player = players[row];
            check_index(idx);
            if (seen[idx]) throw py::value_error("Duplicate environment index");
            seen[idx] = true;
            if (player < 0 || player >= NUM_PLAYERS || offsets[row] < 0
                || offsets[row + 1] <= offsets[row] || offsets[row + 1] > actions.shape(0)
                || choices[row] < 0 || choices[row] >= offsets[row + 1] - offsets[row])
                throw py::value_error("Invalid player or candidate choice");
            check_capacity(idx, 1);
        }
        const auto* observations = obs.data();
        const auto* candidates = actions.data();
        const auto* labels = auxiliary.data();
        {
            py::gil_scoped_release release;
            for (size_t row = 0; row < rows; ++row) {
                const size_t selected = size_t(offsets[row]) + size_t(choices[row]);
                episodes_[indices[row]][players[row]].append(observations + row * STATE_DIM,
                    candidates + selected * ACTION_DIM, labels + row * 5);
                ++counts_[indices[row]];
            }
        }
    }

    void record_block(int idx, const ByteArray& obs, const std::vector<int>& players,
                      const ByteArray& actions, const LabelArray& auxiliary) {
        check_index(idx);
        const size_t rows = players.size();
        check_arrays(obs, actions, auxiliary, rows);
        if (actions.shape(0) != (py::ssize_t)rows)
            throw py::value_error("Rule observations and actions must have identical lengths");
        for (int player : players)
            if (player < 0 || player >= NUM_PLAYERS) throw py::value_error("Invalid player");
        check_capacity(idx, rows);
        for (size_t row = 0; row < rows; ++row)
            episodes_[idx][players[row]].append(obs.data() + row * STATE_DIM,
                actions.data() + row * ACTION_DIM, auxiliary.data() + row * 5);
        counts_[idx] += rows;
    }

    py::list finish(int idx, const std::vector<double>& payoffs, const py::object& rewards) {
        check_index(idx);
        if (payoffs.size() != NUM_PLAYERS) throw py::value_error("One payoff per player is required");
        std::vector<std::vector<double>> step_rewards;
        if (!rewards.is_none()) {
            step_rewards = rewards.cast<std::vector<std::vector<double>>>();
            if (step_rewards.size() != NUM_PLAYERS)
                throw py::value_error("One reward sequence per player is required");
            for (int player = 0; player < NUM_PLAYERS; ++player)
                if (step_rewards[player].size() != episodes_[idx][player].size())
                    throw py::value_error("Step rewards and recorded actions must have identical lengths");
        }
        py::list result;
        for (int player = 0; player < NUM_PLAYERS; ++player) {
            auto* columns = new Columns(std::move(episodes_[idx][player]));
            py::capsule owner(columns, [](void* ptr) { delete static_cast<Columns*>(ptr); });
            const py::ssize_t rows = columns->size();
            py::dict role;
            role["state"] = py::array_t<int8_t>({rows, py::ssize_t(STATE_DIM)},
                {py::ssize_t(STATE_DIM), py::ssize_t(1)}, columns->state.data(), owner);
            role["action"] = py::array_t<int8_t>({rows, py::ssize_t(ACTION_DIM)},
                {py::ssize_t(ACTION_DIM), py::ssize_t(1)}, columns->action.data(), owner);
            role["aux_target"] = py::array_t<int64_t>({rows, py::ssize_t(5)},
                {py::ssize_t(5*sizeof(int64_t)), py::ssize_t(sizeof(int64_t))}, columns->auxiliary.data(), owner);
            py::array_t<float> target(rows), episode_return(rows);
            py::array_t<bool> done(rows);
            double total = payoffs[player], reward_sum = 0.;
            for (py::ssize_t row = rows; row-- > 0;) {
                if (!step_rewards.empty()) total += step_rewards[player][row];
                target.mutable_data()[row] = (float)total;
                done.mutable_data()[row] = false;
                episode_return.mutable_data()[row] = 0.f;
            }
            if (!step_rewards.empty())
                for (double reward : step_rewards[player]) reward_sum += reward;
            if (rows) {
                done.mutable_data()[rows - 1] = true;
                episode_return.mutable_data()[rows - 1] = (float)(payoffs[player] + reward_sum);
            }
            role["target"] = target;
            role["done"] = done;
            role["episode_return"] = episode_return;
            result.append(role);
        }
        clear(idx);
        return result;
    }
};
}  // namespace a3dizhu_v12
