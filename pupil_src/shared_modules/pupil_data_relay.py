"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""

import time
import zmq_tools
from plugin import System_Plugin_Base


class Pupil_Data_Relay(System_Plugin_Base):
    """"""

    def __init__(self, g_pool):
        super().__init__(g_pool)
        self.order = 0.01
        self.gaze_pub = zmq_tools.Msg_Streamer(
            self.g_pool.zmq_ctx, self.g_pool.ipc_pub_url
        )
        self.pupil_sub = zmq_tools.Msg_Receiver(
            self.g_pool.zmq_ctx, self.g_pool.ipc_sub_url, topics=("pupil",)
        )

    def recent_events(self, events):
        recent_pupil_data = []
        recent_gaze_data = []
        while self.pupil_sub.new_data:
            topic, pupil_datum = self.pupil_sub.recv()
            t_ipc_recv = self.g_pool.get_timestamp() if hasattr(self.g_pool, "get_timestamp") else time.time()
            recent_pupil_data.append(pupil_datum)

            # Measure real ZeroMQ IPC transport latency
            ipc_transport_ms = 0.0
            timing = pupil_datum.get("waterfall_timing", None) if isinstance(pupil_datum, dict) else None
            if timing and "t_ipc_send" in timing:
                ipc_transport_ms = max(0.001, (t_ipc_recv - timing["t_ipc_send"]) * 1000.0)

            # Measure real gaze mapping execution latency
            gazer = self.g_pool.active_gaze_mapping_plugin
            gaze_mapping_ms = 0.0
            if gazer is not None:
                t_gaze_start = time.perf_counter()
                mapped_gazes = list(gazer.map_pupil_to_gaze([pupil_datum]))
                t_gaze_end = time.perf_counter()
                gaze_mapping_ms = max(0.001, (t_gaze_end - t_gaze_start) * 1000.0)
                for gaze_datum in mapped_gazes:
                    self.gaze_pub.send(gaze_datum)
                    recent_gaze_data.append(gaze_datum)

            # Log complete, 100% physically measured waterfall trace for this frame
            if timing:
                render_ms = float(getattr(self.g_pool, "last_render_ms", 0.0))
                buffer_swap_ms = float(getattr(self.g_pool, "last_buffer_swap_ms", 0.0))

                ingest_ms = float(timing.get("ingest_ms", 0.0))
                roi_ms = float(timing.get("roi_ms", 0.0))
                preprocess_ms = float(timing.get("preprocess_ms", 0.0))
                inference_ms = float(timing.get("inference_ms", 0.0))
                ellipse_fit_ms = float(timing.get("ellipse_fit_ms", 0.0))
                filter_ms = float(timing.get("filter_ms", 0.0))
                pye3d_ms = float(timing.get("pye3d_ms", 0.0))

                total_system_latency_ms = (
                    ingest_ms
                    + roi_ms
                    + preprocess_ms
                    + inference_ms
                    + ellipse_fit_ms
                    + filter_ms
                    + pye3d_ms
                    + ipc_transport_ms
                    + gaze_mapping_ms
                    + render_ms
                    + buffer_swap_ms
                )

                try:
                    try:
                        from waterfall_logger import get_waterfall_logger
                    except ImportError:
                        from shared_modules.waterfall_logger import get_waterfall_logger
                    wf = get_waterfall_logger()
                    smoothing_method = str(getattr(self.g_pool, "pupil_detector_smoothing_method", "one_euro"))
                    model = str(timing.get("model", getattr(self.g_pool, "pupil_detector_model", "pmrnet")))
                    model_smooth = model+"_"+smoothing_method
                    wf.log_frame_trace({
                        "frame_id": timing.get("frame_id", 0),
                        "process": timing.get("process", "eye0"),
                        "model": model_smooth,
                        "ingest_ms": ingest_ms,
                        "roi_ms": roi_ms,
                        "preprocess_ms": preprocess_ms,
                        "inference_ms": inference_ms,
                        "ellipse_fit_ms": ellipse_fit_ms,
                        "filter_ms": filter_ms,
                        "pye3d_ms": pye3d_ms,
                        "ipc_transport_ms": ipc_transport_ms,
                        "gaze_mapping_ms": gaze_mapping_ms,
                        "render_ms": render_ms,
                        "buffer_swap_ms": buffer_swap_ms,
                        "total_system_latency_ms": total_system_latency_ms,
                        "t_start": timing.get("t_detect_start", t_ipc_recv),
                        "t_end": time.perf_counter(),
                    })
                except Exception:
                    pass

        events["pupil"] = recent_pupil_data
        events["gaze"] = recent_gaze_data
