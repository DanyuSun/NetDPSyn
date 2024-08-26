import copy
import math

from lib_dpsyn.records_update import RecordUpdate


class UpdateConfig:
    def __init__(self, domain, num_records, update_config):
        self.update = RecordUpdate(domain, num_records)

        self.alpha = update_config["alpha"]
        self.alpha_update_method = update_config["alpha_update_method"]
        self.update_method = update_config["update_method"]
        self.threshold = update_config["threshold"]

    def update_alpha(self, iteration):
        # remain unchanged
        if self.alpha_update_method == "U1":
            self.alpha = self.alpha

        # specific exponential decay 1
        elif self.alpha_update_method == "U2":
            self.alpha *= 0.98

        # specific exponential decay 2
        elif self.alpha_update_method == "U3":
            if iteration < 100:
                self.alpha *= 0.98
            else:
                self.alpha *= 0.99

        # step decay
        elif self.alpha_update_method == "U4":
            self.alpha = 1.0 * 0.84 ** (iteration // 20)

        # exponential decay
        elif self.alpha_update_method == "U5":
            self.alpha = math.exp(- 0.008 * iteration)

        # 1/t decay / linear decay
        elif self.alpha_update_method == "U6":
            self.alpha = 1.0 / (1.0 + 0.02 * iteration)

        # square root decay
        elif self.alpha_update_method == "U7":
            self.alpha = 1.0 / math.sqrt(0.12 * iteration + 1.0)

        # fix 1.0
        elif self.alpha_update_method == "U8":
            self.alpha = 1.0

        # fix 0.2
        elif self.alpha_update_method == "U9":
            self.alpha = 0.2

        # fix 0.3
        elif self.alpha_update_method == "U10":
            self.alpha = 0.3

        # fix 0.5
        elif self.alpha_update_method == "U11":
            self.alpha = 0.5

        # fix 0.1
        elif self.alpha_update_method == "U12":
            self.alpha = 0.1

        else:
            raise Exception("invalid alpha update method")

    def update_order(self, iteration, views, iterate_keys):
        for key in iterate_keys:
            self.update.update_records_before(views[key], key, iteration, mute=True)

        sort_error_tracker = self.update.error_tracker.sort_values(by="%s-before" % (iteration,), ascending=False)

        return list(sort_error_tracker.index)

    def update_records(self, original_view, view_key, iteration):
        view = copy.deepcopy(original_view)
        self.update.update_records_before(view, view_key, iteration, mute=True)

        self.update.update_records_main(view, self.alpha)
        self.update.determine_throw_indices()
        self.update.handle_zero_cells(view)

        # main difference
        if self.update_method == "S1":
            self.update.complete_partial_ratio(view, 0.0)
        elif self.update_method == "S2":
            self.update.complete_partial_ratio(view, 1.0)
        elif self.update_method == "S3":
            self.update.complete_partial_ratio(view, 0.5)
        elif self.update_method == "S4":
            if iteration % 2 == 0:
                self.update.complete_partial_ratio(view, 0.0)
            else:
                self.update.complete_partial_ratio(view, 1.0)
        elif self.update_method == "S5":
            if iteration % 2 == 0:
                self.update.complete_partial_ratio(view, 0.5)
            else:
                self.update.complete_partial_ratio(view, 1.0)
        elif self.update_method == "S6":
            if iteration % 2 == 0:
                self.update.complete_partial_ratio(view, 0.5)
            else:
                self.update.complete_partial_ratio(view, 0.0)
        else:
            raise Exception("invalid update method")

        self.update.update_records_after(view, view_key, iteration)
