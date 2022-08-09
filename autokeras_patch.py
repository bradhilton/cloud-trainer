import autokeras as ak
from autokeras.graph import Graph
from IPython.display import clear_output
import keras
from keras.callbacks import Callback
from keras_tuner import HyperParameters, Oracle
from keras_tuner.engine import hyperparameters
from keras_tuner.engine.trial import Trial
import logging
from proxy_auto_model import ProxyAutoModel
from multiprocessing import Manager, Process
from multiprocessing.managers import DictProxy
import os
import sys
import tensorflow as tf
from tensorflow.python.framework.errors_impl import NotFoundError
import time
from typing import Any, cast, Optional, TextIO, Union


def apply() -> None:
    global _did_apply_patch
    if not _did_apply_patch:
        _did_apply_patch = True
        ak.AutoModel.__new__ = _AutoModel__new__patch
        ak.AutoModel.__init__ = _AutoModel__init__patch
        ak.AutoModel.fit = _AutoModel__fit_patch
        Graph._compile_keras_model = _Graph_compile_keras_model_patch


_did_apply_patch = False
_AutoModel__new__ = ak.AutoModel.__new__
_AutoModel__init__ = ak.AutoModel.__init__
_AutoModel_fit = ak.AutoModel.fit
_Graph_compile_keras_model = Graph._compile_keras_model


def _AutoModel__new__patch(
    cls: type[ak.AutoModel],
    *args: Any,
    host: Union[str, tuple[str, int], None] = None,
    **kwargs: Any,
) -> ak.AutoModel:
    if host:
        return cast(
            ak.AutoModel,
            ProxyAutoModel(
                host=host if isinstance(host, str) else f"{host[0]}:{host[1]}",
                args=args,
                kwargs=kwargs,
            ),
        )

    return _AutoModel__new__(cls)


def _AutoModel__init__patch(self: ak.AutoModel, *args: Any, **kwargs: Any) -> None:
    setattr(self, "_init_args_and_kwargs", (args, kwargs.copy()))

    compile_choices = [
        (key, kwargs.pop(key, None)) for key in ("optimizer", "learning_rate")
    ]

    _AutoModel__init__(self, *args, **kwargs)

    setattr(
        self.tuner.hypermodel,
        "_compile_choices",
        {
            name: (
                choice
                if isinstance(choice, hyperparameters.Choice)
                else hyperparameters.Choice(name, [choice])
            )
            for name, choice in compile_choices
            if choice is not None
        },
    )


class _NumberPlaceholder:
    def __format__(self, format_spec: str) -> str:
        return "*" * len((0).__format__(format_spec))


def _AutoModel__fit_patch(
    self: ak.AutoModel,
    *args: Any,
    num_tuners: int = 1,
    **kwargs: Any,
) -> Any:
    if num_tuners <= 1:
        return _AutoModel_fit(self, *args, **kwargs)

    epochs = kwargs.get("epochs", 1_000)
    train_length = len(kwargs["y"])
    batch_size = kwargs.get("batch_size", 32)
    steps_per_epoch = kwargs.get("steps_per_epoch", (train_length // batch_size) + 1)

    manager = Manager()
    tuner_state: dict[int, "DictProxy[str, Any]"] = {
        id: manager.dict(
            trial="?",
            next_trial=1,
            epoch=0,
            batch=0,
            loss=_NumberPlaceholder(),
            acc=_NumberPlaceholder(),
            val_loss=_NumberPlaceholder(),
            val_acc=_NumberPlaceholder(),
        )
        for id in range(num_tuners)
    }
    init_args, init_kwargs = getattr(self, "_init_args_and_kwargs")
    chief_process = Process(
        target=_chief_process,
        kwargs={
            "init_args": init_args,
            "init_kwargs": init_kwargs,
        },
    )
    tuner_processes = [
        Process(
            target=_tuner_process,
            kwargs={
                "tuner_id": id,
                "tuner_state": tuner_state[id],
                "init_args": init_args,
                "init_kwargs": init_kwargs,
                "fit_args": args,
                "fit_kwargs": kwargs,
            },
        )
        for id in range(num_tuners)
    ]
    print("Starting chief process...")
    chief_process.start()
    for process in tuner_processes:
        print("Starting tuner process...")
        process.start()
    oracle: Oracle = self.tuner.oracle
    best_trial: Optional[Trial] = None

    col_width = 18

    def format_value(val: Any) -> str:
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return "{:.5g}".format(val)
        else:
            val_str = str(val)
            if len(val_str) > col_width:
                val_str = val_str[: col_width - 3] + "..."
            return val_str

    tensorflow_logger = tf.get_logger()
    tensorflow_log_level = tensorflow_logger.getEffectiveLevel()
    tensorflow_logger.setLevel(logging.WARN)

    while True:
        try:
            time.sleep(0.5)

            try:
                oracle.reload()
            except RuntimeError:
                try:
                    oracle = ak.AutoModel(*init_args, **init_kwargs).tuner.oracle
                except RuntimeError:
                    continue
            except NotFoundError:
                pass

            for key, trial in oracle.ongoing_trials.items():
                tuner_state[int(key.split("tuner")[1])]["next_trial"] = (
                    int(trial.trial_id) + 1
                )

            new_best_trial: Optional[Trial] = next(iter(oracle.get_best_trials()), None)
            if new_best_trial and (
                not best_trial or best_trial.trial_id != new_best_trial.trial_id
            ):
                best_trial = new_best_trial
                print(f"\033[2J")
                print(f"\033[1;1H")
                clear_output(wait=True)
                print(f"Best Trial So Far: #{int(best_trial.trial_id) + 1}")
                print(f"Score: {best_trial.score:.4f}\n")
                template = "{{0:{0}}}|{{1}}".format(col_width)
                print(template.format("Best Value So Far", "Hyperparameter"))
                for hp, value in best_trial.hyperparameters.values.items():
                    print(
                        template.format(
                            format_value(value),
                            hp,
                        )
                    )
                print("")

            print(
                "  ".join(
                    _tuner_state_description(
                        state, oracle.max_trials or 100, epochs, steps_per_epoch
                    )
                    for state in tuner_state.values()
                ),
                end="\r",
            )
        except BaseException as exception:
            tensorflow_logger.setLevel(tensorflow_log_level)
            chief_process.terminate()
            for process in tuner_processes:
                process.terminate()
            raise exception


def _tuner_state_description(
    state: "DictProxy[str, Any]", trials: int, epochs: int, steps_per_epoch: int
) -> str:
    progress = f"{state['epoch']:>{len(str(epochs))}}/{epochs} {_progress_bar(state['batch'], steps_per_epoch)}"
    metric = lambda metric: f"{metric}: {state[metric]:.4f}"
    trial_num = f"#{state['trial']}"
    return (
        f"🤖 Trial {trial_num:>{len(str(trials)) + 1}}: "
        + " - ".join(
            [
                progress,
                metric("loss"),
                metric("acc"),
                f"{metric('val_loss')} (best: {state.get('score', _NumberPlaceholder()):.4f})",
                metric("val_acc"),
            ]
        )
    ).replace(" ", "\u00a0")


def _progress_bar(progress: float, total: float) -> str:
    bar_length = 20
    filled_length = int(bar_length * progress / total)
    bar = (
        "=" * filled_length
        + ("." if progress <= 0 else ("" if progress >= total else ">"))
        + "." * (bar_length - filled_length - 1)
    )
    return f"[{bar}]"


def _get_writeable_file(path: str) -> TextIO:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, "w")


def _chief_process(*, init_args: tuple, init_kwargs: dict) -> None:
    os.environ["KERASTUNER_TUNER_ID"] = f"chief"
    os.environ["KERASTUNER_ORACLE_IP"] = "localhost"
    os.environ["KERASTUNER_ORACLE_PORT"] = str(5450)
    sys.stdout = _get_writeable_file("logs/stdout/chief.txt")
    sys.stderr = _get_writeable_file("logs/stderr/chief.txt")
    print("Starting chief...")
    ak.AutoModel(*init_args, **init_kwargs)


def _tuner_process(
    *,
    tuner_id: int,
    tuner_state: "DictProxy[str, Any]",
    init_args: tuple,
    init_kwargs: dict,
    fit_args: tuple,
    fit_kwargs: dict,
) -> None:
    global _shared_tuner_state
    _shared_tuner_state = cast(dict, tuner_state)
    os.environ["KERASTUNER_TUNER_ID"] = f"tuner{tuner_id}"
    os.environ["KERASTUNER_ORACLE_IP"] = "localhost"
    os.environ["KERASTUNER_ORACLE_PORT"] = str(5450)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(tuner_id)
    sys.stdout = _get_writeable_file(f"logs/stdout/tuner{tuner_id}.txt")
    sys.stderr = _get_writeable_file(f"logs/stderr/tuner{tuner_id}.txt")
    print("\nStarting tuner...")
    ak.AutoModel(*init_args, **init_kwargs).fit(
        *fit_args,
        callbacks=fit_kwargs.pop("callbacks", []) + [TunerCallback()],
        **fit_kwargs,
    )


_shared_tuner_state: dict[str, Any] = {}


class TunerCallback(Callback):
    def on_train_begin(self, logs: dict) -> None:
        _shared_tuner_state["trial"] = _shared_tuner_state["next_trial"]
        _shared_tuner_state["epoch"] = 1
        _shared_tuner_state["batch"] = 0
        _shared_tuner_state["loss"] = _NumberPlaceholder()
        _shared_tuner_state["acc"] = _NumberPlaceholder()
        _shared_tuner_state["val_loss"] = _NumberPlaceholder()
        _shared_tuner_state["val_acc"] = _NumberPlaceholder()
        _shared_tuner_state.pop("score", None)

    def on_epoch_begin(self, epoch: int, logs: dict) -> None:
        _shared_tuner_state["epoch"] = epoch + 1
        _shared_tuner_state["batch"] = 0

    def on_test_end(self, logs: dict) -> None:
        _shared_tuner_state["val_loss"] = logs["loss"]
        _shared_tuner_state["val_acc"] = logs["accuracy"]
        _shared_tuner_state["score"] = min(
            _shared_tuner_state.get("score", float("inf")),
            _shared_tuner_state["val_loss"],
        )

    def on_train_batch_end(self, batch: int, logs: dict):
        _shared_tuner_state["batch"] = batch + 1
        _shared_tuner_state["loss"] = logs["loss"]
        _shared_tuner_state["acc"] = logs["accuracy"]


def _Graph_compile_keras_model_patch(
    self: Graph, hp: HyperParameters, model: keras.Model
) -> keras.Model:
    __Choice = hp.Choice

    def Choice(name: str, values: list, *args: Any, **kwargs: Any) -> Any:
        choice: Optional[hyperparameters.Choice] = getattr(
            self, "_compile_choices", {}
        ).get(name)
        if choice:
            values = choice.values
            kwargs["ordered"] = choice.ordered
            kwargs["default"] = choice.default
        return __Choice(name, values, *args, **kwargs)

    hp.Choice = Choice

    return _Graph_compile_keras_model(self, hp, model)
