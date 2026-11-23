"""A separate, scrollable editor for a fixed diagonal cost matrix."""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from util import get_ik_dof_count, get_rotation_joint_indices

if TYPE_CHECKING:
    import tkinter as tk
    from fairmotion.core.motion import Pose


LINK_COLOR = "#66268f"
JOINT_COLOR = "#bf75e6"
HOVER_COLOR = "#e53935"


class DiagonalWeightEditor:
    """Edit a draft and commit in place so the solver keeps the same W reference."""

    def __init__(self, W: np.ndarray, preview_pose: "Pose") -> None:
        skel = preview_pose.skel
        joint_names = tuple(joint.name for joint in skel.joints)
        rotation_indices = get_rotation_joint_indices(skel)
        num_dofs = get_ik_dof_count(skel)
        if W.shape != (num_dofs, num_dofs):
            raise ValueError("W shape must match root translation and non-leaf joint rotations")
        if np.any(W != np.diag(np.diag(W))):
            raise ValueError("The diagonal editor requires a diagonal matrix")
        self.W = W
        self.joint_names = joint_names
        self.parent_indices = tuple(
            None if joint.parent_joint is None else skel.get_index_joint(joint.parent_joint.name)
            for joint in skel.joints
        )
        # Snapshot a readable pose; zero joint rotations can collapse BVH bones onto one axis.
        self.skeleton_positions = np.array([
            preview_pose.get_transform(name, local=False)[:3, 3] for name in joint_names
        ])
        self.row_names = ("Hips translation",) + tuple(
            f"{joint_names[j]} rotation" for j in rotation_indices
        )
        self.row_joint_indices = (0,) + rotation_indices
        self.hovered_joint: int | None = None
        self.preview_yaw = np.deg2rad(20.0)
        self.preview_pitch = np.deg2rad(8.0)
        self.drag_position: tuple[int, int] | None = None
        self.window: "tk.Tk | None" = None
        self.entries: list["tk.Entry"] = []
        self.on_close: Callable[[bool], None] | None = None

    @property
    def is_open(self) -> bool:
        return self.window is not None

    def open(self, on_close: Callable[[bool], None]) -> None:
        """Open once, with no change to W until Apply succeeds."""
        import tkinter as tk
        from tkinter import ttk

        if self.window is not None:
            self.window.lift()
            return
        self.on_close = on_close
        self.hovered_joint = None
        self.drag_position = None
        root = self.window = tk.Tk()
        root.withdraw()
        root.title("Edit W | Fixed diagonal weights")
        width = min(1080, max(940, root.winfo_screenwidth() - 80))
        height = min(760, max(440, root.winfo_screenheight() - 120))
        x = max(0, (root.winfo_screenwidth() - width) // 2)
        y = max(0, (root.winfo_screenheight() - height) // 2 - 30)
        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(940, 440)
        root.protocol("WM_DELETE_WINDOW", self.cancel)
        root.bind("<Escape>", lambda event: self.cancel())
        root.bind("<Return>", lambda event: self.cancel()
                  if event.widget is self.cancel_button else self.apply())

        header = ttk.Frame(root, padding=(16, 12, 16, 8))
        header.pack(fill="x")
        ttk.Label(header, text=f"Diagonal W: {self.W.shape[0]} positive values",
                  font=("Segoe UI", 12, "bold")).pack(anchor="w")
        ttk.Label(header, text="Larger weight = greater motion cost. Changes apply to this session.").pack(anchor="w", pady=(5, 0))
        ttk.Label(header, text="Hips translation: world axes. Joint rotations: local axes.").pack(anchor="w")

        # Pack the footer first so it stays visible when the window is resized.
        footer = ttk.Frame(root, padding=(16, 8, 16, 12))
        footer.pack(side="bottom", fill="x")
        self.error_text = tk.StringVar(master=root)
        ttk.Label(footer, textvariable=self.error_text, foreground="#b42318",
                  wraplength=600).pack(fill="x", pady=(0, 8))
        buttons = ttk.Frame(footer)
        buttons.pack(fill="x")
        self.apply_button = ttk.Button(buttons, text="Apply", command=self.apply)
        self.apply_button.pack(side="right")
        self.cancel_button = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.cancel_button.pack(side="right", padx=(0, 8))

        content = ttk.Frame(root)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=1, minsize=300)
        content.columnconfigure(1, weight=2, minsize=620)
        content.rowconfigure(0, weight=1)
        preview = ttk.Frame(content, padding=(16, 4, 8, 0))
        preview.grid(row=0, column=0, sticky="nsew")
        ttk.Label(preview, text="Skeleton (first frame)", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        ttk.Label(preview, text="Hover a row or joint. Drag to rotate.").pack(anchor="w", pady=(4, 8))
        self.hover_name = tk.StringVar(master=root, value="Joint: -")
        ttk.Label(preview, textvariable=self.hover_name, font=("Segoe UI", 11, "bold"),
                  foreground=HOVER_COLOR).pack(side="bottom", fill="x", pady=(8, 0))
        self.preview_canvas = tk.Canvas(preview, background="#faf8fc", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.bind("<Configure>", lambda event: self.draw_skeleton())
        self.preview_canvas.bind("<Motion>", self.hover_preview)
        self.preview_canvas.bind("<Leave>", lambda event: self.set_hovered_joint(None))
        self.preview_canvas.bind("<ButtonPress-1>", self.begin_preview_drag)
        self.preview_canvas.bind("<B1-Motion>", self.rotate_preview)
        self.preview_canvas.bind("<ButtonRelease-1>", self.end_preview_drag)

        table = ttk.Frame(content, padding=(8, 0, 16, 0))
        table.grid(row=0, column=1, sticky="nsew")
        column_header = ttk.Frame(table)
        column_header.pack(fill="x", padx=(0, 18))
        for column, label in enumerate(("Component", "x", "y", "z")):
            column_header.columnconfigure(column, weight=2 if column == 0 else 1,
                                          minsize=250 if column == 0 else 100)
            ttk.Label(column_header, text=label).grid(row=0, column=column, sticky="w", padx=4, pady=5)
        self.canvas = tk.Canvas(table, highlightthickness=0)
        scrollbar = ttk.Scrollbar(table, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        body = ttk.Frame(self.canvas)
        body_id = self.canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", lambda event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(body_id, width=event.width))
        root.bind("<MouseWheel>", self.scroll_mousewheel)
        root.bind("<Button-4>", lambda event: self.canvas.yview_scroll(-3, "units"))
        root.bind("<Button-5>", lambda event: self.canvas.yview_scroll(3, "units"))
        for column in range(4):
            body.columnconfigure(column, weight=2 if column == 0 else 1,
                                 minsize=250 if column == 0 else 100)
        self.entries = []
        self.row_labels = []
        diagonal = np.diag(self.W)
        for row, name in enumerate(self.row_names):
            joint_index = self.row_joint_indices[row]
            label = ttk.Label(body, text=name)
            label.grid(row=row, column=0, sticky="ew", padx=4, pady=5)
            self.bind_joint_hover(label, joint_index)
            self.row_labels.append(label)
            for axis in range(3):
                entry = tk.Entry(body, width=12, relief="solid", borderwidth=1)
                entry.insert(0, str(float(diagonal[3 * row + axis])))
                entry.grid(row=row, column=axis + 1, sticky="ew", padx=4, pady=5, ipady=3)
                entry.bind("<FocusIn>", lambda event, widget=entry: self.reveal_entry(widget))
                self.bind_joint_hover(entry, joint_index)
                self.entries.append(entry)
        root.update_idletasks()
        root.deiconify()
        root.lift()
        self.entries[0].focus_set()

    def bind_joint_hover(self, widget: "tk.Widget", joint_index: int) -> None:
        widget.bind("<Enter>", lambda event: self.set_hovered_joint(joint_index))
        widget.bind("<Leave>", lambda event: self.set_hovered_joint(None))

    def set_hovered_joint(self, joint_index: int | None) -> None:
        if self.window is None:
            return
        if self.hovered_joint == joint_index:
            return
        self.hovered_joint = joint_index
        self.hover_name.set("Joint: -" if joint_index is None else f"Joint: {self.joint_names[joint_index]}")
        for row, label in enumerate(self.row_labels):
            label.configure(foreground=HOVER_COLOR if self.row_joint_indices[row] == joint_index else "#202020")
        self.draw_skeleton()

    def draw_skeleton(self) -> None:
        """Project the fixed skeleton snapshot onto a rotatable orthographic canvas."""
        if self.window is None:
            return
        canvas = self.preview_canvas
        width, height = canvas.winfo_width(), canvas.winfo_height()
        if min(width, height) <= 1:
            return
        cy, sy = np.cos(self.preview_yaw), np.sin(self.preview_yaw)
        cp, sp = np.cos(self.preview_pitch), np.sin(self.preview_pitch)
        right = np.array([cy, 0, -sy])
        up = np.array([-sy * sp, cp, -cy * sp])
        depth_axis = np.array([sy * cp, sp, cy * cp])
        projection = self.skeleton_positions @ np.stack([right, up], axis=1)
        lower, upper = projection.min(axis=0), projection.max(axis=0)
        extent = np.maximum(upper - lower, 1e-6)
        scale = min(max(width - 48, 1) / extent[0], max(height - 48, 1) / extent[1])
        projection = (projection - (lower + upper) / 2) * scale
        self.preview_points = np.column_stack([width / 2 + projection[:, 0], height / 2 - projection[:, 1]])
        self.preview_depths = self.skeleton_positions @ depth_axis
        canvas.delete("all")
        for child, parent in enumerate(self.parent_indices):
            if parent is not None:
                canvas.create_line(*self.preview_points[parent], *self.preview_points[child],
                                   fill=LINK_COLOR, width=3, capstyle="round")
        self.joint_items = {}
        order = [int(j) for j in np.argsort(self.preview_depths) if j != self.hovered_joint]
        if self.hovered_joint is not None:
            order.append(self.hovered_joint)
        for j in order:
            x, y = self.preview_points[j]
            selected = j == self.hovered_joint
            radius = 8 if selected else 5
            self.joint_items[j] = canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                fill=HOVER_COLOR if selected else JOINT_COLOR,
                outline="white", width=1,
            )

    def hover_preview(self, event: "tk.Event") -> None:
        distances = np.linalg.norm(self.preview_points - [event.x, event.y], axis=1)
        nearest = int(np.argmin(distances))
        self.set_hovered_joint(nearest if distances[nearest] <= 11 else None)

    def begin_preview_drag(self, event: "tk.Event") -> None:
        self.drag_position = (event.x, event.y)
        self.set_hovered_joint(None)

    def rotate_preview(self, event: "tk.Event") -> None:
        if self.drag_position is None:
            return
        x, y = self.drag_position
        self.preview_yaw += (event.x - x) * 0.008
        self.preview_pitch = float(np.clip(self.preview_pitch + (event.y - y) * 0.008, -1.2, 1.2))
        self.drag_position = (event.x, event.y)
        self.draw_skeleton()

    def end_preview_drag(self, event: "tk.Event") -> None:
        self.drag_position = None
        self.hover_preview(event)

    def scroll_mousewheel(self, event: "tk.Event") -> str:
        if event.widget is self.preview_canvas:
            return "break"
        if event.delta:
            steps = -int(event.delta / 120) if abs(event.delta) >= 120 else (-1 if event.delta > 0 else 1)
            self.canvas.yview_scroll(steps * 3, "units")
        return "break"

    def reveal_entry(self, entry: "tk.Entry") -> None:
        """Keep the focused entry visible while tabbing through a long table."""
        bounds = self.canvas.bbox("all")
        if not bounds or bounds[3] <= 0:
            return
        top = self.canvas.canvasy(0)
        bottom = top + self.canvas.winfo_height()
        entry_top, entry_bottom = entry.winfo_y(), entry.winfo_y() + entry.winfo_height()
        if entry_top < top:
            self.canvas.yview_moveto(entry_top / bounds[3])
        elif entry_bottom > bottom:
            self.canvas.yview_moveto((entry_bottom - self.canvas.winfo_height()) / bounds[3])

    def apply(self) -> None:
        """Validate every field before changing the live matrix."""
        values = []
        for index, entry in enumerate(self.entries):
            try:
                value = float(entry.get())
                if not np.isfinite(value) or value <= 0:
                    raise ValueError
            except ValueError:
                name, axis = self.row_names[index // 3], "xyz"[index % 3]
                self.error_text.set(f"{name} / {axis}: enter a finite number greater than 0.")
                entry.focus_set()
                entry.selection_range(0, "end")
                self.reveal_entry(entry)
                return
            values.append(value)
        np.fill_diagonal(self.W, values)
        self.close(applied=True)

    def cancel(self) -> None:
        self.close(applied=False)

    def close(self, applied: bool = False) -> None:
        if self.window is None:
            return
        window, self.window = self.window, None
        window.destroy()
        self.entries = []
        callback, self.on_close = self.on_close, None
        if callback is not None:
            callback(applied)

    def poll(self) -> None:
        """Service Tk events from the OpenGL idle callback on the same thread."""
        if self.window is not None:
            self.window.update()
