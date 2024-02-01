from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn

class ES(torch.nn.Module):

    def __init__(self,):
        super(ES, self).__init__()
        self.mini_crane = 1/30
        gru_hidden_size = 32
        attention_hidden_size_1 = 64
        attention_hidden_size_2 = 64

        self.emb_1 = nn.Linear(5, gru_hidden_size)
        self.emb_2 = nn.Linear(4, gru_hidden_size)
        self.emb_3 = nn.Linear(3, attention_hidden_size_1)
        self.emb_4 = nn.Linear(3, attention_hidden_size_1)
        self.emb_5 = nn.Linear(4, attention_hidden_size_1)
        self.emb_6 = nn.Linear(4, attention_hidden_size_1)
        self.gru1 = nn.GRU(input_size=gru_hidden_size, hidden_size=gru_hidden_size, batch_first=True)
        self.gru2 = nn.GRU(input_size=gru_hidden_size, hidden_size=gru_hidden_size, batch_first=True)

        self.attention_1 = nn.MultiheadAttention(embed_dim=attention_hidden_size_1, num_heads=1, batch_first=True)
        self.attention_2 = nn.MultiheadAttention(embed_dim=attention_hidden_size_1, num_heads=1, batch_first=True)

        self.linear = nn.Linear(2 * gru_hidden_size, attention_hidden_size_1)
        self.linear_att = nn.Linear(attention_hidden_size_1, attention_hidden_size_2)
        self.linear_final = nn.Linear(attention_hidden_size_2, 2)
        self.activate = torch.tanh
        self.epsilon = 1e-6

    def cope_state(self, state):
        filter_berthed = (state[:, 1] > -self.epsilon) & (state[:, 3] > 0)
        filter_waiting = (state[:, 1] < 0) & (state[:, 5] <= self.epsilon)
        filter_future = state[:, 1] < self.epsilon
        id = torch.arange(state.shape[0]).to(filter_berthed.device)[filter_waiting]
        seq1 = torch.cat((state[filter_berthed, :2], state[filter_berthed, 4:5] / state[filter_berthed, 2:3],
                          state[filter_berthed, 7:8], state[filter_berthed, 2:3] + state[filter_berthed, 7:8]),
                         dim=1).to(filter_berthed.device)

        inf_mask = torch.isinf(seq1)
        seq1[inf_mask] = 999

        if seq1.shape[0] == 0:
            seq1 = torch.zeros((1, 5)).to(filter_berthed.device)
        seq2 = torch.cat(
            (
                state[filter_future, 3:4], state[filter_future, 4:5], state[filter_future, 6:7],
                state[filter_future, 5:6]),
            dim=1)
        seq3 = torch.cat((state[filter_waiting, 3:4], state[filter_waiting, 4:5], state[filter_waiting, 6:7]), dim=1)

        return seq1, seq2, seq3, id

    def get_pos(self, state, length):
        filter_berthed = (state[:, 1] > -self.epsilon) & (state[:, 3] > 0)
        if torch.sum(filter_berthed) == 0:
            return torch.tensor(((0, 1, 0, 1),)).float().to(filter_berthed.device), torch.tensor((1,)).float().to(
                filter_berthed.device)
        max_length = torch.max(length, state[filter_berthed, 3])
        pos = state[filter_berthed, :2]
        pos[:, 0] -= 0.1 * max_length
        pos[:, 1] += 0.1 * max_length
        flattened_array = pos.flatten()
        final_array = torch.cat([flattened_array, torch.tensor([0, 1]).to(state.device)])
        sorted_pos = torch.sort(final_array).values.reshape(-1, 2)
        lengtn_pos = sorted_pos[:, 1] - sorted_pos[:, 0]

        crane_pos = torch.cat((state[filter_berthed, 7:8], state[filter_berthed, 7:8] + state[filter_berthed, 2:3] - self.mini_crane), dim=1)
        flattened_array = crane_pos.flatten()
        final_array = torch.cat([flattened_array, torch.tensor([0, 1 - self.mini_crane + self.epsilon]).to(state.device)])
        sorted_crane = torch.sort(final_array).values.reshape(-1, 2)
        crane_length = sorted_crane[:, 1] - sorted_crane[:, 0]
        filter_crane = crane_length>1.5*self.mini_crane

        seq4 = torch.cat((sorted_pos, sorted_crane), dim=1)
        return seq4[filter_crane], lengtn_pos.to(filter_berthed.device)[filter_crane]

    def forward(self, state):
        state = state.squeeze()
        seq1, seq2, seq3, id = self.cope_state(state)
        s1, s2 = self.emb_1(seq1), self.emb_2(seq2)
        if s1.dim()<3:
            s1,s2 = s1.unsqueeze(0),s2.unsqueeze(0)
        global_info1, _ = self.gru1(s1)
        global_info2, _ = self.gru2(s2)
        global_info1,global_info2 = global_info1.squeeze(0),global_info2.squeeze(0)

        merged_global_info = torch.cat((global_info1[-1, :], global_info2[-1, :]), dim=-1)

        query = self.linear(merged_global_info).unsqueeze(0)
        key = self.emb_3(seq3)
        value = self.emb_3(seq3)
        query, key, value = query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0)

        result, attention_weights = self.attention_1(query, key, value)
        try:
            max_index = torch.argmax(attention_weights)
        except:
            print(seq3)
            print(state)
        length = seq3[max_index, 0]
        max_crane = seq3[max_index, 2]

        seq4, lengtn_pos = self.get_pos(state, length)
        new_query = self.linear_att(result)
        new_port = seq4[lengtn_pos >= length]
        if new_port.shape[0] <= 0:
            return np.array([-1, -1, -1, -1, -1])
        new_key, new_value = self.emb_5(new_port).unsqueeze(0), self.emb_6(new_port).unsqueeze(0)
        final_result, final_attention_weights = self.attention_2(new_query, new_key, new_value)
        final_attention_weights = final_attention_weights.reshape(-1, 1)
        pos = torch.sigmoid(self.linear_final(final_result).squeeze())
        p_f = pos[0].item()
        c_f = pos[1].item()
        final_max_index = torch.argmax(final_attention_weights).item()
        pos_all_dis = new_port[final_max_index, 1] - new_port[final_max_index, 0]
        p1 = (pos_all_dis - length) * p_f + new_port[final_max_index, 0]
        p2 = p1 + length
        crane_all_number = new_port[final_max_index, 3] - new_port[final_max_index, 2]
        if crane_all_number > max_crane:
            c1 = (crane_all_number - max_crane) * c_f + new_port[final_max_index, 2]
            c2 = c1 + max_crane
        else:
            c1 = new_port[final_max_index, 2]
            c2 = new_port[final_max_index, 3]


        return np.array((id[max_index].item(), p1.item(), c1.item(), c2.item(),
                         torch.sigmoid(final_attention_weights[final_max_index]).item()))

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.cpu().data.numpy().flatten().shape[0]
        return count

    def es_params(self):
        return [(k, v) for k, v in zip(self.state_dict().keys(),
                                       self.state_dict().values())]

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=10)
                nn.init.constant_(m.bias, 0.)


