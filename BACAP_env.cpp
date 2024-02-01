#include <vector>
#include <algorithm>
#include <array>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iterator>
#include <tuple>




class  BerthOptimizationEnvironment {
public:
	static const int state_number = 30;
	static constexpr double epsilon = 1e-5;
	std::vector<std::tuple<double, double, int, int>> solution;
	std::vector<std::array<double, 2>> vessel_pos;
	std::vector<std::tuple<int, double>> arrival_vessel;
	std::vector<double> vessel_length;
	std::vector<double> vessel_capacity;
	std::vector<double> vessel_arrival_time;
	std::vector<int> vessel_max_crane;
	std::vector<int> vessel_crane_number;
	std::vector<std::vector<bool>> vessel_have_crane;
	std::vector<double> crane_pos;
	std::vector<bool> crane_work;
	double bay_length = 3000;
	double crane_work_efficiency = 1;
	double vessel_secure_ratio = 0.1;
	double crane_secure_distance = 0.001;
	double wait_time = 0.0;
	double time = 0.0;
	int step_id = 0;
	int can_arrange_vessel_number = 0;
	int num_cranes;
	int num_vessel;
	int finish_vessel = 0;



	BerthOptimizationEnvironment(const char* file) {
		std::string file_str(file);
		load(file_str);  
	}


	void load(std::string file) {
		std::ifstream infile(file);
		std::string line;
		int line_number = 0;
		while (std::getline(infile, line)) {
			std::istringstream iss(line);
			std::vector<std::string> row_data;
			std::string value;
			while (std::getline(iss, value, '\t')) {
				row_data.push_back(value);
			}
			if (line_number == 0) {
				num_cranes = std::stoi(row_data[4]);
			}
			vessel_arrival_time.push_back(std::stod(row_data[0]));
			vessel_capacity.push_back(std::stod(row_data[1]));
			vessel_length.push_back(std::stod(row_data[2]));
			vessel_max_crane.push_back(std::stoi(row_data[3]));
			vessel_pos.push_back({ -1.0, -1.0 });
			vessel_crane_number.push_back(0);
			vessel_have_crane.push_back(std::vector<bool>(num_cranes, false));
			solution.push_back(std::make_tuple(0, 1, 0, 1));
			line_number++;
		}
		for (int i = 0; i < num_cranes; ++i) {
			crane_work.push_back(false);
			crane_pos.push_back(i * bay_length / num_cranes);
		}
		num_vessel = static_cast<int>(vessel_arrival_time.size());
	}
	bool vessel_put(int id, double pos) {
		if (pos < -epsilon || pos + vessel_length[id] > bay_length) {
			return false;
		}

		if (vessel_arrival_time[id] - time > epsilon) {
			return false;
		}

		if (vessel_pos[id][0] >= -epsilon) {
			return false;
		}

		if (vessel_capacity[id] <= epsilon) {
			return false;
		}

		if (!check_vessel_cross(id, pos)) {
			return false;
		}

		vessel_pos[id][0] = pos;
		vessel_pos[id][1] = pos + vessel_length[id];

		return true;
	}

	bool check_vessel_cross(int id, double pos) {
		double a = pos; 
		double b = pos + vessel_length[id];

		for (int i = 0; i < num_vessel; ++i) {
			if (vessel_pos[i][0] < -epsilon || i == id) continue;  

			double p = vessel_pos[i][0];
			double l = vessel_length[i];
			double lmax = vessel_secure_ratio * std::max(l, vessel_length[id]); 

			if (!(b <= p - lmax + epsilon) && !(a >= vessel_pos[i][1] + lmax - epsilon)) {
				return false;
			}
		}

		return true;
	}

	bool crane_put(int id, int vessel_id) {
		double pos = vessel_pos[vessel_id][0];
		if (vessel_max_crane[vessel_id] == vessel_crane_number[vessel_id] || pos < -epsilon) {
			return false;
		}
		if (crane_work[id]) {
			return false;
		}
		if (std::abs(pos - crane_pos[id]) <= epsilon) {
			vessel_have_crane[vessel_id][id] = true;
			vessel_crane_number[vessel_id]++;
			crane_work[id] = true;
			return true;
		}
		int id_next = id;
		if (pos < crane_pos[id]) {
			for (int i = id - 1; i >= 0; i--) {
				if (crane_work[i] && crane_pos[i] > pos + epsilon) {
					return false;
				}
				if (crane_pos[i] <= pos + epsilon) break;
				id_next = i;
			}
			vessel_have_crane[vessel_id][id] = true;
			vessel_crane_number[vessel_id]++;
			crane_work[id] = true;
			for (int i = id; i >= id_next; i--) {
				crane_pos[i] = pos;
			}
		}
		else {
			for (int i = id + 1; i < num_cranes; i++) {
				if (crane_work[i] && crane_pos[i] < pos - epsilon) {
					return false;
				}
				if (crane_pos[i] >= pos - epsilon) break;
				id_next = i;
			}
			vessel_have_crane[vessel_id][id] = true;
			vessel_crane_number[vessel_id]++;
			crane_work[id] = true;
			for (int i = id; i <= id_next; i++) {
				crane_pos[i] = pos;
			}

		}

		return true;
	}


	void preempt_crane() {
		for (const auto& vessel : arrival_vessel) {
			if (std::get<1>(vessel) <= 1) {
				break;
			}
			put_vessel_priority(std::get<0>(vessel));


		}
	}

	void sort_arrival_vessel() {
		std::sort(arrival_vessel.begin(), arrival_vessel.end(), [](const std::tuple<int, double>& left, const std::tuple<int, double>& right) {
			return std::get<1>(left) > std::get<1>(right);
			});
	}

	void printVesselPos() {
		std::cout << "vessel_pos:" << std::endl;
		for (const auto& pos : vessel_pos) {
			std::cout << pos[0] << ", " << pos[1] << std::endl;
		}
	}

	void printVesselCapacity() {
		std::cout << "vessel_capacity:" << std::endl;
		for (const auto& capacity : vessel_capacity) {
			std::cout << capacity << std::endl;
		}
	}
	bool step_time(double t) {
		int num_waiting_vessels = 0;
		int max_vessel_id = num_vessel;
		for (int i = 0; i < num_vessel; i++) {

			if (vessel_capacity[i] > epsilon) {
				if (vessel_arrival_time[i] <= (time + epsilon)) {
					num_waiting_vessels++;
				}
				else {
					max_vessel_id = i;
					break;
				}
			}
		}

		wait_time += num_waiting_vessels * t;
		bool flag = false;
		for (int i = 0; i < max_vessel_id; i++) {
			if (vessel_pos[i][0] >= -epsilon && vessel_crane_number[i] > 0) {
				vessel_capacity[i] -= crane_work_efficiency * vessel_crane_number[i] * t;

				if (vessel_capacity[i] <= epsilon) {
					vessel_capacity[i] = 0;
					vessel_pos[i][0] = -1;
					vessel_pos[i][1] = -1;
					vessel_crane_number[i] = 0;
					for (int j = 0; j < num_cranes; j++) {
						if (vessel_have_crane[i][j]) {
							crane_work[j] = false;
							vessel_have_crane[i][j] = false;
						}
					}
					arrival_vessel.erase(
						std::remove_if(arrival_vessel.begin(), arrival_vessel.end(),
							[i](const std::tuple<int, double>& vessel) {
								return std::get<0>(vessel) == i;
							}),
						arrival_vessel.end());
					finish_vessel++;
				}
			}
		}

		time += t;

		while (step_id < num_vessel && vessel_arrival_time[step_id] <= time + epsilon) {
			arrival_vessel.push_back(std::make_tuple(step_id, std::get<1>(solution[step_id])));
			can_arrange_vessel_number += 1;
			step_id++;
			flag = true;
		}
		if (flag) {
			sort_arrival_vessel();
		}
		return finish_vessel == num_vessel;
	}
	bool has_unberthed_ships() {
		for (const auto& vessel : arrival_vessel) {
			int index = std::get<0>(vessel); 
			if (vessel_pos[index][0] < -epsilon) { 
				return true;
			}
		}
		return false;
	}
	std::pair<bool, double> next_vessel_time() {
		bool flag = std::any_of(crane_work.begin(), crane_work.end(), [](bool work) {return work; });
		double t1 = std::numeric_limits<double>::max(); 
		if (flag) {
			for (int i = 0; i < step_id; i++) {
				if (vessel_pos[i][0] >= -epsilon && vessel_crane_number[i] > 0 && vessel_capacity[i] > epsilon) {
					double time_to_finish = vessel_capacity[i] / (crane_work_efficiency * vessel_crane_number[i]);
					if (time_to_finish < t1) {
						t1 = time_to_finish;
					}
				}
			}
		}

		if (step_id < num_vessel) {
			double t2 = vessel_arrival_time[step_id] - time;
			if (flag) {

				if (t1 < t2) {
					return std::make_pair(false, t1);
				}
				else {
					return std::make_pair(true, t2);
				}
			}
			else {
				return std::make_pair(true, t2);
			}
		}
		else {
			if (flag) {
				return std::make_pair(false, t1);
			}
			else {
				return std::make_pair(true, 0.0);
			}
		}




	}


	std::array<std::array<double, 9>, state_number> get_state() {

		std::array<std::array<double, 9>, state_number> state;

		int count = 0;
		int id = step_id;

		for (int i = 0; i < std::min(state_number, static_cast<int>(arrival_vessel.size())); i++) {
			int index = std::get<0>(arrival_vessel[i]); 

			state[count][0] = (vessel_pos[index][0] < -epsilon) ? -1 : (vessel_pos[index][0] / bay_length); 
			state[count][1] = (vessel_pos[index][1] < -epsilon) ? -1 : (vessel_pos[index][1] / bay_length); 
			state[count][2] = 1.0 * vessel_crane_number[index] / num_cranes; 
			state[count][3] = vessel_length[index] / bay_length; 
			state[count][4] = 1.0 * vessel_capacity[index] / num_cranes; 
			state[count][5] = std::max(0.0, vessel_arrival_time[index] - time); 
			state[count][6] = 1.0 * (vessel_max_crane[index]) / num_cranes; 
			state[count][7] = 1.0 * std::get<2>(solution[index]) / num_cranes; 
            state[count][8] = index; 
			count++;
		}

		while (count < state_number && id < num_vessel) {
			state[count][0] = (vessel_pos[id][0] < -epsilon) ? -1 : (vessel_pos[id][0] / bay_length);
			state[count][1] = (vessel_pos[id][1] < -epsilon) ? -1 : (vessel_pos[id][1] / bay_length);
			state[count][2] = 1.0 * vessel_crane_number[id] / num_cranes;
			state[count][3] = vessel_length[id] / bay_length;
			state[count][4] = 1.0 * vessel_capacity[id] / num_cranes;
			state[count][5] = vessel_arrival_time[id] - time;
			state[count][6] = 1.0 * (vessel_max_crane[id]) / num_cranes;
			state[count][7] = 1.0 * std::get<2>(solution[id]) / num_cranes; 
            state[count][8] = id;
			count++;
			id++;
		}

		while (count < state_number) {
			state[count] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
			count++;
		}

		return state;
	}
	std::tuple<std::array<std::array<double, 9>, state_number>, double, bool>
		step(std::array<double, 5> action) {
		double all_step_time = wait_time;
		int arrival_vessel_num = std::min(state_number, static_cast<int>(arrival_vessel.size()));
		int id = static_cast<int>(action[0]); 
		if ((id >= arrival_vessel.size()) || (id < 0 || std::get<1>(arrival_vessel[id])>1)) {
			bool flag = true;
			double times;
			do
			{
				std::pair<bool, double> result = next_vessel_time();
				std::tie(flag, times) = result;
				preempt_crane();
				if (step_time(times)) {
					return std::make_tuple(get_state(), all_step_time - wait_time, true);
				}
				preempt_crane();
			} while (!has_unberthed_ships());
			return std::make_tuple(get_state(), all_step_time - wait_time, false);	
		}

		int can_arrange_vessel = 0;

		for (int i = 0; i < arrival_vessel_num; i++) {

			double pro = std::get<1>(arrival_vessel[i]);
			if (pro <= 1) {
				std::get<1>(arrival_vessel[i]) = pro * 0.9;
				can_arrange_vessel += 1;
			}

		}

		double pos = action[1];
		pos *= bay_length;
		int crane_id = static_cast<int>(std::round(action[2] * num_cranes));
		int crane_end = static_cast<int>(std::round(action[3] * num_cranes));
		int crane_num = crane_end - crane_id + 1;
		double priority = action[4];

        if (crane_num <=0){
            bool flag = true;
			double times;
			do
			{
				std::pair<bool, double> result = next_vessel_time();
				std::tie(flag, times) = result;
				preempt_crane();
				if (step_time(times)) {
					return std::make_tuple(get_state(), all_step_time - wait_time, true);
				}
				preempt_crane();
			} while (!has_unberthed_ships());
			return std::make_tuple(get_state(), all_step_time - wait_time, false);	
        }
        
		int index = std::get<0>(arrival_vessel[id]);
		solution[index] = { pos,priority,crane_id,crane_num };
		put_vessel_priority(index);
		if (vessel_pos[index][1] > -epsilon) {
			std::get<1>(arrival_vessel[id]) = priority + 1;
			can_arrange_vessel_number -= 1;
			sort_arrival_vessel();
		}
		else {
			std::get<1>(arrival_vessel[id]) = 1;
			sort_arrival_vessel();
			bool flag = true;
			double times;
			do
			{
				std::pair<bool, double> result = next_vessel_time();
				std::tie(flag, times) = result;
				preempt_crane();
				if (step_time(times)) {
					return std::make_tuple(get_state(), all_step_time - wait_time, true);
				}
				preempt_crane();
			} while (!has_unberthed_ships());
			return std::make_tuple(get_state(), all_step_time - wait_time, false);	
		}


		if (has_unberthed_ships()){
			return std::make_tuple(get_state(), all_step_time - wait_time, false);	
		}
		else{
			bool flag = true;
			double times;
			do
			{
				std::pair<bool, double> result = next_vessel_time();
				std::tie(flag, times) = result;
				preempt_crane();
				if (step_time(times)) {
					return std::make_tuple(get_state(), all_step_time - wait_time, true);
				}
				preempt_crane();
			} while (!has_unberthed_ships());
			return std::make_tuple(get_state(), all_step_time - wait_time, finish_vessel == num_vessel);	

		}
	}
	bool put_vessel_priority(int vessel_id) {
		if (vessel_capacity[vessel_id] <= epsilon) {
			return false;
		}

		if (vessel_crane_number[vessel_id] == vessel_max_crane[vessel_id]) {
			return false;
		}

		double pos = std::get<0>(solution[vessel_id]);
		double priority = std::get<1>(solution[vessel_id]);
		int crane_id = std::get<2>(solution[vessel_id]);
		int crane_num = std::get<3>(solution[vessel_id]);
		if (priority <= 1) {
			if (vessel_pos[vessel_id][0] < -epsilon) {
				if (vessel_put(vessel_id, pos)) {
					priority += 1;
				}
				else {
					pos = -1;
					priority = 1;
				}

			}
			bool first_crane = true;
			if (vessel_pos[vessel_id][0] >= -epsilon && vessel_crane_number[vessel_id] < vessel_max_crane[vessel_id]) {
				for (int i = crane_id; i < std::min(num_cranes, crane_id + crane_num); i++) {
					if (first_crane && crane_put(i, vessel_id)) {
						crane_id = i;
						first_crane = false;
					}
					if (vessel_crane_number[vessel_id] == vessel_max_crane[vessel_id]) {
						break;
					}
				}
			}
			crane_num = vessel_crane_number[vessel_id];
			if (crane_num == 0) {
				crane_id = -1;
			}
			solution[vessel_id] = { pos,priority,crane_id,crane_num };
		}
		else {
			if (vessel_crane_number[vessel_id] == 0) {
				int left_f = std::min(num_cranes - 1, static_cast<int>(std::round(pos / bay_length * num_cranes)));
				int right_f = left_f;
				while ((left_f >= 0) || (right_f < num_cranes)) {
					if (left_f >= 0) {
						crane_put(left_f, vessel_id);
						left_f -= 1;
						if (vessel_crane_number[vessel_id] > 0) {
							crane_id = left_f + 1;
							crane_num = vessel_crane_number[vessel_id];
							break;
						}
					}

					if (right_f < num_cranes) {
						crane_put(right_f, vessel_id);
						right_f += 1;
						if (vessel_crane_number[vessel_id] > 0) {
							crane_id = right_f - 1;
							crane_num = vessel_crane_number[vessel_id];
							break;
						}
					}
				}

			}
	
			if (crane_num > 0) {
				bool left_flag = crane_id - 1 > 0;
				bool right_flag = (crane_id + crane_num) <= num_cranes - 1;
				while (left_flag || right_flag) {
					if (left_flag) {
						left_flag = crane_put(crane_id - 1, vessel_id);
						if (left_flag) {
							crane_id -= 1;
							if (crane_id - 1 < 0) {
								left_flag = false;
							}
						}
					}
					if (right_flag) {
						right_flag = crane_put(crane_id + crane_num, vessel_id);
						if (right_flag) {
							crane_num += 1;
							if (crane_id + crane_num > num_cranes - 1) {
								right_flag = false;
							}
						}
					}
				}
			}
			solution[vessel_id] = { pos,priority,crane_id,crane_num };
		}


		return true;
	}




	bool is_all_done() {
		int count = static_cast<int>(std::count_if(vessel_capacity.begin(), vessel_capacity.end(), [](double capacity) {
			return capacity <= epsilon;
			}));
		return num_vessel == count;
	}

	std::array<std::array<double, 9>, state_number> reset() {
		time = 0.0;
		wait_time = 0.0;
		double t = vessel_arrival_time[0];
		step_time(t);
		return get_state();
	}

	void printState() {
		const std::array<std::array<double, 9>, state_number>& state = get_state();
		std::cout << "State: " << time << std::endl;
		for (const auto& vessel : state) {
			for (const auto& value : vessel) {
				std::cout << value << " ";
			}
			std::cout << std::endl;
		}
	}
	void printSolution() {
		std::cout << "Solution:" << std::endl;
		for (const auto& sol : solution) {
			std::cout << "Pos: " << std::get<0>(sol)
				<< ", Priority: " << std::get<1>(sol)
				<< ", Crane ID: " << std::get<2>(sol)
				<< ", Crane Num: " << std::get<3>(sol) << std::endl;
		}
	}

};




struct Array2D {
    double data[BerthOptimizationEnvironment::state_number][9];
};


struct StepResult {
    Array2D state;
    double reward;
    bool done;
};

extern "C" {
    __declspec(dllexport) void* create_env(const char* file) {
        return new BerthOptimizationEnvironment(file);
    }

    __declspec(dllexport) void delete_env(BerthOptimizationEnvironment* env) {
        delete env;
    }

    __declspec(dllexport) void step(BerthOptimizationEnvironment* env, std::array<double, 5> action, StepResult* result) {
        auto [state, reward, done] = env->step(action);
        for (int i = 0; i < BerthOptimizationEnvironment::state_number; ++i) {
            for (int j = 0; j < 9; ++j) {
                result->state.data[i][j] = state[i][j];
            }
        }
        result->reward = reward;
        result->done = done;
    }

    __declspec(dllexport) void reset(BerthOptimizationEnvironment* env, Array2D* arr) {
        auto state = env->reset();
        for (int i = 0; i < BerthOptimizationEnvironment::state_number; ++i) {
            for (int j = 0; j < 9; ++j) {
                arr->data[i][j] = state[i][j];
            }
        }
    }
}