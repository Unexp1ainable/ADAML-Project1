from helpers import *
from pathlib import Path
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import numpy as np
import pandas as pd



class TimeLagOptimizer:
    def __init__(self, step: int, upper_lag_boundary: int, out_folder: None | str = None) -> None:
        self.step = step
        self.upper_lag_boundary = upper_lag_boundary
        self.out_folder = out_folder
        self.results = None
        self.boundaries = [
            "% Iron Feed",
            "Ore Pulp Density",
            "Flotation Column 01 Level",
            "Flotation Column 02 Level",
            "Flotation Column 03 Level",
            "Flotation Column 04 Level",
            "Flotation Column 05 Level",
            "Flotation Column 06 Level",
            "Flotation Column 07 Level",
            # "y"
        ]
        self.csv_header = "train_loss,valid_loss\n"

    def _optimize(self, train_x: pd.DataFrame, train_y: pd.DataFrame, valid_x: pd.DataFrame, valid_y: pd.DataFrame):
        raise NotImplementedError()

    @staticmethod
    def evaluate(y_true, y_pred):
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def should_stop(scores: list[float]) -> bool:
        N_TO_STOP = 10
        if len(scores) < N_TO_STOP:
            return False

        last = scores[-N_TO_STOP]
        for score in scores[-N_TO_STOP+1:]:
            if last - score > 0:
                return False
            last = score

        return True

    def loggerReset(self, path: str):
        if self.out_folder:
            Path(f"./{self.out_folder}").mkdir(parents=True, exist_ok=True)
            with open(f"{self.out_folder}/{path}", "w") as file:
                file.write(self.csv_header)

    def log(self, path: str, msg: str):
        if self.out_folder:
            with open(f"{self.out_folder}/{path}", "a") as file:
                file.write(msg + "\n")

    
    def run(self, train_x: pd.DataFrame, train_y: pd.DataFrame, valid_x: pd.DataFrame, valid_y: pd.DataFrame):
        res = self._optimize(train_x, train_y, valid_x, valid_y)
        self.results = res
        return res



class VerticalStairsOptimizer(TimeLagOptimizer):
    def _optimize(self, train_x: pd.DataFrame, train_y: pd.DataFrame, valid_x: pd.DataFrame, valid_y: pd.DataFrame):
        n_components = train_x.shape[1]

        optimal_working_copy = train_x.copy()
        optimal_y_working_copy = train_y.copy()
        optimal_lags = []

        for column_begin, column_end in zip(self.boundaries[:-1], self.boundaries[1:]):
            id = f"{column_begin} - {column_end}"
            self.loggerReset(f"{id}.csv")
            print(f"Processing \"{id}\"")

            train_scores = []
            valid_scores = []
            optimal_lag = 0
            working_copy = optimal_working_copy.copy()
            y_working_copy = optimal_y_working_copy.copy()
            early_stop = False

            for i in tqdm(range(0, self.upper_lag_boundary, self.step)):
                # make model
                model = PLSRegression(n_components=n_components)
                model.fit(working_copy, y_working_copy)

                pred_train = model.predict(working_copy).flatten()
                pred_valid = model.predict(valid_x).flatten()

                # evaluate performance
                train_score = self.evaluate(y_working_copy, pred_train)
                valid_score = self.evaluate(valid_y, pred_valid)
                train_scores.append(train_score)
                valid_scores.append(valid_score)
                self.log(id, f"{train_score}, {valid_score}")

                shift(working_copy, y_working_copy, self.step, column_begin, column_end)

                # determine early stopping
                if self.should_stop(valid_scores):
                    print("Early stopping engaged")
                    optimal_lag = np.argmin(valid_scores) * self.step
                    early_stop = True
                    break

            if not early_stop:
                optimal_lag = np.argmin(valid_scores) * self.step

            print(f"Optimum for \"{id}\" = {optimal_lag} s")
            optimal_lags.append(optimal_lag)
            shift(working_copy, y_working_copy, self.step, column_begin, column_end)

        return optimal_lags



def switch(dir):
    if dir == 1:
        return -1
    return 1

def applyShifts(df_x, df_y, shifts, all_boundaries):
    for i, (column_begin, column_end) in enumerate(all_boundaries):
        shift(df_x, df_y, shifts[i], column_begin, column_end)

class DynamicOptimizer(TimeLagOptimizer):
    def __init__(self, step: int, upper_lag_boundary: int, out_folder: str | None = None) -> None:
        super().__init__(step, upper_lag_boundary, out_folder)
        
        all_boundaries = list(zip(self.boundaries[:-1], self.boundaries[1:]))
        self.csv_header = ",".join(map(lambda x: "-".join([x[0],x[1]]), all_boundaries)) + "\n"
        self.valid_loss_file = "losses_valid.csv"
        self.train_loss_file = "losses_train.csv"
        self.results_file = "results.csv"

    def loggerReset(self):
        super().loggerReset(self.valid_loss_file)
        super().loggerReset(self.train_loss_file)
        super().loggerReset(self.results_file)


    def logResults(self, results):
        with open(f"{self.out_folder}/{self.results_file}", "a") as file:
            file.write(",".join(map(str, results)) + "\n")

    def logLossesTrain(self, losses):
        with open(f"{self.out_folder}/{self.train_loss_file}", "a") as file:
            file.write(",".join(map(str, losses)) + "\n")

    def logLossesValid(self, losses):
        with open(f"{self.out_folder}/{self.valid_loss_file}", "a") as file:
            file.write(",".join(map(str, losses)) + "\n")

    def _optimize(self, train_x: pd.DataFrame, train_y: pd.DataFrame, valid_x: pd.DataFrame, valid_y: pd.DataFrame):
        n_components = train_x.shape[1]
        self.loggerReset()

        all_boundaries = list(zip(self.boundaries[:-1], self.boundaries[1:]))
        results_count = len(all_boundaries)
        directions = np.ones((results_count))
        steps = np.array([self.step for _ in range(results_count)], dtype=np.float64)

        # let's assume that one flotation phase should take 15 minutes
        # https://www.chem.mtu.edu/chem_eng/faculty/kawatra/Flotation_Fundamentals.pdf
        phase_length = 3
        preliminary_results = np.array([(results_count-i) * phase_length for i in range(results_count)])

        last_valid_scores = np.zeros((results_count))
        train_scores = np.zeros((results_count))
        valid_scores = np.zeros((results_count))

        # initialize preliminary values
        for i, (column_begin, column_end) in enumerate(all_boundaries):
            x_working_copy = train_x.copy()
            y_working_copy = train_y.copy()
            shift(x_working_copy, y_working_copy, preliminary_results[i], column_begin, column_end)

            model = PLSRegression(n_components=n_components)
            model.fit(x_working_copy, y_working_copy)

            pred_valid = model.predict(valid_x).flatten()

            # evaluate performance
            last_valid_scores[i] = self.evaluate(valid_y, pred_valid)

        # preliminary_results += steps
        while True:
            x_progress = train_x.copy()
            y_progress = train_y.copy()
            applyShifts(x_progress, y_progress, preliminary_results, all_boundaries)

            # determine model improvements over potential lag shifts
            for i, (column_begin, column_end) in enumerate(all_boundaries):
                id = f"{column_begin} - {column_end}"

                x_working_copy = x_progress.copy()
                y_working_copy = y_progress.copy()

                shift(x_working_copy, y_working_copy, int(preliminary_results[i]+self.step*directions[i]), column_begin, column_end)

                model = PLSRegression(n_components=n_components)
                model.fit(x_working_copy, y_working_copy)

                tmp_valid_x = valid_x.copy()
                tmp_valid_y = valid_y.copy()
                s = preliminary_results
                s[i] += self.step * directions[i]
                applyShifts(tmp_valid_x, tmp_valid_y, s, all_boundaries)

                pred_train = model.predict(x_working_copy).flatten()
                pred_valid = model.predict(tmp_valid_x).flatten()

                # evaluate performance
                train_scores[i] = self.evaluate(y_working_copy, pred_train)
                valid_scores[i] = self.evaluate(tmp_valid_y, pred_valid)
                directions[i] = directions[i] if last_valid_scores[i] - valid_scores[i] > 0 else switch(directions[i])

            self.logLossesValid(valid_scores)
            self.logLossesTrain(train_scores)
            self.logResults(preliminary_results)

            # scale to best improvement
            # diff = last_valid_scores-valid_scores
            # if np.abs(diff).sum() != 0:
            #     steps = np.abs(np.round(diff / diff.sum() * self.step,0))
            # else:
            #     break

            steps *= directions
            preliminary_results += steps.astype(np.int64)
            print(np.min(valid_scores))
            if (abs(np.min(last_valid_scores) - np.min(valid_scores)) < 0.00000001):
                break
            last_valid_scores = valid_scores.copy()

        return preliminary_results





class DynamicOptimizerOneAtATime(TimeLagOptimizer):
    def __init__(self, step: int, upper_lag_boundary: int, out_folder: str | None = None) -> None:
        super().__init__(step, upper_lag_boundary, out_folder)
        
        all_boundaries = list(zip(self.boundaries[:-1], self.boundaries[1:]))
        self.csv_header = ",".join(map(lambda x: "-".join([x[0],x[1]]), all_boundaries)) + "\n"
        self.valid_loss_file = "losses_valid.csv"
        self.train_loss_file = "losses_train.csv"
        self.results_file = "results.csv"

    def loggerReset(self):
        super().loggerReset(self.valid_loss_file)
        super().loggerReset(self.train_loss_file)
        super().loggerReset(self.results_file)


    def logResults(self, results):
        with open(f"{self.out_folder}/{self.results_file}", "a") as file:
            file.write(",".join(map(str, results)) + "\n")

    def logLossesTrain(self, losses):
        with open(f"{self.out_folder}/{self.train_loss_file}", "a") as file:
            file.write(",".join(map(str, losses)) + "\n")

    def logLossesValid(self, losses):
        with open(f"{self.out_folder}/{self.valid_loss_file}", "a") as file:
            file.write(",".join(map(str, losses)) + "\n")

    def _optimize(self, train_x: pd.DataFrame, train_y: pd.DataFrame, valid_x: pd.DataFrame, valid_y: pd.DataFrame):
        n_components = train_x.shape[1]
        self.loggerReset()

        all_boundaries = list(zip(self.boundaries[:-1], self.boundaries[1:]))
        results_count = len(all_boundaries)
        directions = np.ones((results_count))

        # let's assume that one flotation phase should take 3 hours
        phase_length = 0
        preliminary_results = np.array([(results_count-i) * phase_length for i in range(results_count)])

        last_valid_score = 0

        # initialize preliminary values
        for i, (column_begin, column_end) in enumerate(all_boundaries):
            x_working_copy = train_x.copy()
            y_working_copy = train_y.copy()
            shift(x_working_copy, y_working_copy, preliminary_results[i], column_begin, column_end)

            model = PLSRegression(n_components=n_components)
            model.fit(x_working_copy, y_working_copy)

            pred_valid = model.predict(valid_x).flatten()

            # evaluate performance
            last_valid_score = self.evaluate(valid_y, pred_valid)

        # preliminary_results += steps
        x_progress = train_x.copy()
        y_progress = train_y.copy()
        turns = 0
        while turns < 50:
            # applyShifts(x_progress, y_progress, preliminary_results, all_boundaries)

            # determine model improvements over potential lag shifts
            for i, (column_begin, column_end) in enumerate(all_boundaries):
                if (preliminary_results[i]+self.step*directions[i]) < 0:
                    directions[i] = switch(directions[i])
                    continue

                x_working_copy = x_progress.copy()
                y_working_copy = y_progress.copy()

                shift(x_working_copy, y_working_copy, int(preliminary_results[i]+self.step*directions[i]), column_begin, column_end)

                model = PLSRegression(n_components=n_components)
                model.fit(x_working_copy, y_working_copy)

                tmp_valid_x = valid_x.copy()
                tmp_valid_y = valid_y.copy()
                s = preliminary_results.copy()
                s[i] += self.step * directions[i]
                applyShifts(tmp_valid_x, tmp_valid_y, s, all_boundaries)

                pred_train = model.predict(x_working_copy).flatten()
                pred_valid = model.predict(tmp_valid_x).flatten()

                # evaluate performance
                train_score = self.evaluate(y_working_copy, pred_train)
                valid_score = self.evaluate(tmp_valid_y, pred_valid)

                # if model improved, keep going, otherwise turn back
                if last_valid_score - valid_score < 0:
                    # turn back
                    directions[i] = switch(directions[i])
                    turns += 1
                else:
                    turns = 0
                    # make step
                    preliminary_results[i] += self.step*directions[i]
                    print(valid_score)
                    last_valid_score = valid_score
                    x_progress = x_working_copy
                    y_progress = y_working_copy

                self.logLossesValid([valid_score])
                self.logLossesTrain([train_score])
                self.logResults(preliminary_results)

            # scale to best improvement
            # diff = last_valid_scores-valid_scores
            # if np.abs(diff).sum() != 0:
            #     steps = np.abs(np.round(diff / diff.sum() * self.step,0))
            # else:
            #     break

        return preliminary_results


