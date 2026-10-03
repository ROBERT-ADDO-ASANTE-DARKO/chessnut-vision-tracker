import sys
_CLOUDSPACE_SP = '/home/zeus/miniconda3/envs/cloudspace/lib/python3.10/site-packages'
if _CLOUDSPACE_SP not in sys.path:
    sys.path.insert(0, _CLOUDSPACE_SP)

import numpy as np
import math
import supervision as sv
from ultralytics import YOLO, SAM
import pickle
import os
import urllib.request

import cv2
import io
import chess
import chess.pgn
import chess.engine
import mediapipe as mp


# ── Cross-class NMS ───────────────────────────────────────────────────────────
def _cross_class_nms(detections: sv.Detections, iou_threshold: float = 0.5) -> sv.Detections:
    """Remove duplicate detections where two DIFFERENT class boxes cover the same
    physical piece.  YOLO's built-in NMS only suppresses same-class overlaps;
    this second pass suppresses cross-class overlaps (e.g., a bishop also firing
    as a knight on the identical bounding box).

    Strategy: sort boxes by confidence (high→low), greedily keep each box if
    its IoU with every already-accepted box is below ``iou_threshold``.
    """
    if len(detections) == 0:
        return detections

    xyxy  = detections.xyxy                       # (N, 4)
    confs = (detections.confidence
             if detections.confidence is not None
             else np.ones(len(detections)))
    order = np.argsort(-confs)                    # high→low confidence

    keep = []
    for idx in order:
        x1, y1, x2, y2 = xyxy[idx]
        dominated = False
        for k in keep:
            kx1, ky1, kx2, ky2 = xyxy[k]
            # Intersection area
            ix1 = max(x1, kx1); iy1 = max(y1, ky1)
            ix2 = min(x2, kx2); iy2 = min(y2, ky2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            area1 = (x2 - x1) * (y2 - y1)
            area2 = (kx2 - kx1) * (ky2 - ky1)
            iou   = inter / (area1 + area2 - inter + 1e-6)
            if iou >= iou_threshold:
                dominated = True
                break
        if not dominated:
            keep.append(idx)

    return detections[np.array(keep, dtype=int)]


def _square_dist(sq_a: str, sq_b: str) -> float:
    """Euclidean distance in file/rank units between two square names."""
    fa, ra = ord(sq_a[0]) - ord('a'), int(sq_a[1])
    fb, rb = ord(sq_b[0]) - ord('a'), int(sq_b[1])
    return ((fa - fb) ** 2 + (ra - rb) ** 2) ** 0.5


from mediapipe.tasks import python as _mp_tasks
from mediapipe.tasks.python import vision as _mp_vision
from PIL import Image, ImageDraw, ImageFont

# ── Chess-themed color palette (BGR) ──────────────────────────────────────────
C_PANEL_BG     = (26,  24,  21)   # near-black warm background
C_PANEL_BORDER = (62,  58,  53)   # subtle border
C_GREEN        = (86, 150, 118)   # chess.com board green  (#769656)
C_CREAM        = (210, 238, 238)  # chess.com light square (#eeeed2)
C_GOLD         = (40, 190, 255)   # amber highlight
C_TEXT         = (205, 200, 192)  # primary text
C_MUTED        = (110, 105,  98)  # secondary / label text
C_WHITE_IND    = (230, 225, 215)  # white-piece indicator
C_BLACK_IND    = ( 50,  46,  42)  # black-piece indicator
C_EVAL_WHITE   = (228, 224, 218)  # eval bar white fill
C_EVAL_BLACK   = ( 32,  29,  26)  # eval bar black fill
# Move-quality annotation colors (BGR)
C_ANNOT_BRILLIANT = (218, 165,  32)  # gold  — !!
C_ANNOT_GOOD      = ( 80, 180,  80)  # green — !
C_ANNOT_INACCUR   = ( 40, 190, 255)  # amber — ?!
C_ANNOT_MISTAKE   = ( 30, 105, 255)  # orange — ?
C_ANNOT_BLUNDER   = ( 40,  40, 220)  # red   — ??

# ── MediaPipe hand skeleton connections (21-point model) ─────────────────────
# Indices follow the canonical MediaPipe hand landmark numbering:
#   0 = wrist  1-4 = thumb  5-8 = index  9-12 = middle
#   13-16 = ring  17-20 = pinky
_HAND_CONNECTIONS = (
    # Palm rays from wrist + cross-knuckle bar
    (0, 1), (0, 5), (0, 9), (0, 13), (0, 17),
    (5, 9), (9, 13), (13, 17),
    # Thumb
    (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
)
_FINGERTIPS = {4, 8, 12, 16, 20}   # larger dot at each fingertip

# ── Piece symbol map ──────────────────────────────────────────────────────────
PIECE_SYM = {
    'white-king':   '♔', 'white-queen':  '♕', 'white-rook':   '♖',
    'white-bishop': '♗', 'white-knight': '♘', 'white-pawn':   '♙',
    'black-king':   '♚', 'black-queen':  '♛', 'black-rook':   '♜',
    'black-bishop': '♝', 'black-knight': '♞', 'black-pawn':   '♟',
}

PIECE_VALUES = {
    chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
    chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0,
}

# Standard starting piece counts (per side) — used for YOLO-inferred capture estimation
_PIECE_START = {
    chess.PAWN: 8, chess.KNIGHT: 2, chess.BISHOP: 2,
    chess.ROOK: 2, chess.QUEEN:  1, chess.KING:   1,
}
# Map piece-type name string → chess.piece_type int
_PT_FROM_STR = {
    'pawn': chess.PAWN, 'knight': chess.KNIGHT, 'bishop': chess.BISHOP,
    'rook': chess.ROOK, 'queen':  chess.QUEEN,  'king':   chess.KING,
}

# FEN character per YOLO piece-type string  (upper = white, lower = black)
_PIECE_FEN_CHAR = {
    'white-king': 'K', 'white-queen': 'Q', 'white-rook': 'R',
    'white-bishop': 'B', 'white-knight': 'N', 'white-pawn': 'P',
    'black-king': 'k', 'black-queen': 'q', 'black-rook': 'r',
    'black-bishop': 'b', 'black-knight': 'n', 'black-pawn': 'p',
}

# Unicode symbols for captured-piece display (indexed by piece_type, then color)
_CAP_SYM = {
    chess.WHITE: {
        chess.PAWN: '♙', chess.KNIGHT: '♘', chess.BISHOP: '♗',
        chess.ROOK: '♖', chess.QUEEN:  '♕',
    },
    chess.BLACK: {
        chess.PAWN: '♟', chess.KNIGHT: '♞', chess.BISHOP: '♝',
        chess.ROOK: '♜', chess.QUEEN:  '♛',
    },
}

# ── Font cache ────────────────────────────────────────────────────────────────
_FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
_FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
_font_cache: dict = {}

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _font_cache:
        path = _FONT_BOLD if bold else _FONT_PATH
        try:
            _font_cache[key] = ImageFont.truetype(path, size)
        except Exception:
            _font_cache[key] = ImageFont.load_default()
    return _font_cache[key]

# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_rounded_rect(img, pt1, pt2, color, radius=10, thickness=-1):
    """Filled (thickness=-1) or outlined rounded rectangle on an OpenCV image."""
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    r = max(0, min(radius, (x2 - x1) // 2 - 1, (y2 - y1) // 2 - 1))
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
            cv2.circle(img, (cx, cy), r, color, -1)
    else:
        cv2.line(img, (x1+r, y1), (x2-r, y1), color, thickness)
        cv2.line(img, (x1+r, y2), (x2-r, y2), color, thickness)
        cv2.line(img, (x1, y1+r), (x1, y2-r), color, thickness)
        cv2.line(img, (x2, y1+r), (x2, y2-r), color, thickness)
        cv2.ellipse(img, (x1+r, y1+r), (r, r), 180, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y1+r), (r, r), 270, 0, 90, color, thickness)
        cv2.ellipse(img, (x1+r, y2-r), (r, r),  90, 0, 90, color, thickness)
        cv2.ellipse(img, (x2-r, y2-r), (r, r),   0, 0, 90, color, thickness)


def pil_draw(img, draws_fn):
    """
    Convert img to a Pillow canvas, call draws_fn(draw, pil_img), then convert
    back. Batches all Pillow calls for one frame into a single round-trip.
    draws_fn signature: (ImageDraw, Image) -> None
    """
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draws_fn(draw, pil)
    result = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    img[:] = result
    return img


def bgr_to_rgb(bgr):
    """Flip a BGR tuple to RGB for Pillow."""
    return (bgr[2], bgr[1], bgr[0])

'''
def manually_define_board_bbox(frame):
    """
    Manually define the bounding box for the chessboard.
    Args:
        frame (numpy.ndarray): First frame of the video.
    Returns:
        list: Bounding box coordinates [x1, y1, x2, y2].
    """
    print("Please select the chessboard area...")
    board_bbox = cv2.selectROI("Select Chessboard", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select Chessboard")

    # Convert ROI format (x, y, w, h) to bounding box format [x1, y1, x2, y2]
    x, y, w, h = board_bbox
    return [x, y, x + w, y + h]
'''
# Function to detect the chessboard
def detect_chessboard(chessboard_model, frame):
    """
    Detect the chessboard in a frame and return its bounding box.
    Args:
        chessboard_model (YOLO): Trained YOLO model for chessboard detection.
        frame (numpy.ndarray): Video frame.
    Returns:
        list or None: Bounding box coordinates [x1, y1, x2, y2] or None if detection fails.
    """
    results = chessboard_model.predict(frame, conf=0.5, verbose=False)

    if results and results[0].boxes:
        for box in results[0].boxes.xyxy:
            x1, y1, x2, y2 = box.int().tolist()
            # Validate the bounding box (basic sanity check)
            if x2 - x1 > 100 and y2 - y1 > 100:  # Minimum reasonable board size
                return [x1, y1, x2, y2]
    
    # Fallback: manual selection if automatic detection fails
    #print("Automatic chessboard detection failed. Please manually select the chessboard.")
    #board_bbox = cv2.selectROI("Select Chessboard", frame, fromCenter=False, showCrosshair=True)
    #cv2.destroyWindow("Select Chessboard")
    
    # Convert ROI format (x, y, w, h) to [x1, y1, x2, y2]
    #x, y, w, h = board_bbox
    #print("Chessboard bounding box:", board_bbox)
    #return [x, y, x + w, y + h]

class PatternChessboardCornerDetector:
    """
    Finds board outer corners by detecting the inner 7×7 (or smaller) grid of
    square intersections with findChessboardCornersSB, then extrapolating one
    step outward in each direction using the local perspective step vectors.

    Works even when the board edge is partially off-screen, as long as enough
    inner corners are visible.  Falls back gracefully (returns None) when a
    hand or other occlusion covers too many squares.
    """

    def detect_board_corners(self, frame, yolo_bbox=None, min_n=4):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        # Try decreasing grid sizes so partial occlusion is tolerated
        for n in [n for n in [7, 6, 5, 4] if n >= min_n]:
            ret, corners = cv2.findChessboardCornersSB(
                gray, (n, n),
                cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if not ret:
                continue

            grid = corners.reshape(n, n, 2)

            # Average perspective step vectors
            step_r = (grid[:, 1:] - grid[:, :-1]).mean(axis=(0, 1))
            step_d = (grid[1:, :] - grid[:-1, :]).mean(axis=(0, 1))

            TL = grid[0,   0  ] - step_r - step_d
            TR = grid[0,   n-1] + step_r - step_d
            BL = grid[n-1, 0  ] - step_r + step_d
            BR = grid[n-1, n-1] + step_r + step_d

            outer = np.array([TL, TR, BL, BR], dtype=np.float32)

            # Sanity: all corners must be within 25 % of frame bounds
            margin_x, margin_y = W * 0.25, H * 0.25
            if any(cx < -margin_x or cx > W + margin_x or
                   cy < -margin_y or cy > H + margin_y
                   for cx, cy in outer):
                continue

            print(f"  PatternDetector (n={n}): "
                  f"TL={TL.astype(int)} TR={TR.astype(int)} "
                  f"BL={BL.astype(int)} BR={BR.astype(int)}")
            return self._order_corners(outer)

        return None   # all sizes failed

    @staticmethod
    def _order_corners(corners):
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1).ravel()
        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        tr = corners[np.argmin(d)]
        bl = corners[np.argmax(d)]
        return np.array([tl, tr, bl, br], dtype=np.float32)


class LSDChessboardCornerDetector:
    """
    Finds the 4 outer board corners by detecting line segments inside the YOLO
    crop, clustering into horizontal/vertical groups, and intersecting the
    outermost lines. Robust to partial occlusion because board lines extend
    through covered areas — only the outermost edge of each side is needed.
    """

    def __init__(self):
        self.lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)

    def detect_board_corners(self, frame, yolo_bbox=None):
        """
        Detect the 4 outer corners of the chessboard.
        Args:
            frame (numpy.ndarray): Full video frame (BGR).
            yolo_bbox (list, optional): [x1, y1, x2, y2] crop hint from YOLO.
        Returns:
            numpy.ndarray: Shape (4, 2) ordered TL, TR, BL, BR, or None on failure.
        """
        if yolo_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in yolo_bbox]
            pad = int(min(x2 - x1, y2 - y1) * 0.03)
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(frame.shape[1], x2 + pad)
            y2 = min(frame.shape[0], y2 + pad)
            region = frame[y1:y2, x1:x2]
            offset = np.array([x1, y1], dtype=np.float32)
        else:
            region = frame
            offset = np.zeros(2, dtype=np.float32)

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        lines, _, _, _ = self.lsd.detect(gray)

        if lines is None or len(lines) < 4:
            return None

        lines = lines.reshape(-1, 4)  # (N, 4): x1, y1, x2, y2

        # Separate near-horizontal and near-vertical segments by angle
        angles = np.degrees(np.arctan2(lines[:, 3] - lines[:, 1],
                                        lines[:, 2] - lines[:, 0]))
        h_mask = np.abs(angles) < 45
        v_mask = ~h_mask

        h_lines = lines[h_mask]
        v_lines = lines[v_mask]

        if len(h_lines) < 2 or len(v_lines) < 2:
            return None

        # Pick the outermost line on each side by segment midpoint
        h_mid_y = (h_lines[:, 1] + h_lines[:, 3]) / 2
        top_line    = h_lines[h_mid_y.argmin()]
        bottom_line = h_lines[h_mid_y.argmax()]

        v_mid_x = (v_lines[:, 0] + v_lines[:, 2]) / 2
        left_line  = v_lines[v_mid_x.argmin()]
        right_line = v_lines[v_mid_x.argmax()]

        tl = self._intersect(top_line,    left_line)
        tr = self._intersect(top_line,    right_line)
        bl = self._intersect(bottom_line, left_line)
        br = self._intersect(bottom_line, right_line)

        if any(c is None for c in [tl, tr, bl, br]):
            return None

        corners = np.array([tl, tr, bl, br], dtype=np.float32)
        corners += offset  # translate from crop-relative to full-frame coordinates

        # ── Sanity check: reject corners far outside the frame ────────────────
        # The LSD intersect can extrapolate lines to points well beyond the
        # image (e.g. y=-232 when the board has steep perspective). If any
        # corner is more than 25 % of the frame dimension outside the frame
        # bounds, the result is unreliable — return None so the caller falls
        # back to the YOLO bbox rectangle instead.
        fh, fw = frame.shape[:2]
        margin_y = fh * 0.25
        margin_x = fw * 0.25
        for cx, cy in corners:
            if cx < -margin_x or cx > fw + margin_x or cy < -margin_y or cy > fh + margin_y:
                print(f"  LSD corner ({cx:.1f}, {cy:.1f}) out of bounds "
                      f"(frame {fw}×{fh}) — rejecting, will use YOLO bbox fallback")
                return None

        return self._order_corners(corners)

    def _intersect(self, l1, l2):
        """Return the intersection point of two lines defined by endpoint pairs."""
        x1, y1, x2, y2 = l1
        x3, y3, x4, y4 = l2
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    def _order_corners(self, corners):
        """Order 4 points as: top-left, top-right, bottom-left, bottom-right."""
        s = corners.sum(axis=1)
        d = np.diff(corners, axis=1).ravel()
        tl = corners[np.argmin(s)]
        br = corners[np.argmax(s)]
        tr = corners[np.argmin(d)]
        bl = corners[np.argmax(d)]
        return np.array([tl, tr, bl, br], dtype=np.float32)

class PerspectiveTransformer:
    """
    Handles perspective transformation using detected board corners
    """
    def __init__(self, board_corners=None):
        """
        Initialize perspective transformer for chess board.
        Args:
            board_corners (numpy.ndarray): 4x2 array of corner points
        """
        self.board_corners = board_corners
        self.transformation_matrix = None
        self.inverse_matrix = None
        self.board_size = None

    def set_board_corners(self, corners):
        """
        Set the chessboard corners
        Args:
            corners (numpy.ndarray): 4x2 array of corner points
                                    [top-left, top-right, bottom-left, bottom-right]
        """
        self.board_corners = corners
        print(f"Set chessboard corners: {self.board_corners}")

    def estimate_board_perspective(self):
        """
        Estimate board perspective using detected corners
        Returns:
            numpy.ndarray: Perspective transformation matrix
        """
        if self.board_corners is None:
            return None

        # Source points (actual board corners)
        src_points = self.board_corners.astype(np.float32)

        # Calculate board size (use maximum dimension)
        width1 = np.linalg.norm(src_points[0] - src_points[1])
        width2 = np.linalg.norm(src_points[2] - src_points[3])
        height1 = np.linalg.norm(src_points[0] - src_points[2])
        height2 = np.linalg.norm(src_points[1] - src_points[3])
        
        board_size = int(max(width1, width2, height1, height2))
        self.board_size = board_size

        # Define destination points (ideal square board)
        dst_points = np.float32([
            [0, 0],                    # Top-left
            [board_size, 0],           # Top-right
            [0, board_size],           # Bottom-left
            [board_size, board_size]   # Bottom-right
        ])

        # Compute perspective transform
        self.transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        
        return self.transformation_matrix

    def transform_point(self, point):
        """
        Transform a point from frame coordinates to board coordinates
        Args:
            point (tuple): Original point (x, y)
        Returns:
            tuple: Transformed point (x, y)
        """
        if self.transformation_matrix is None:
            return point

        # Convert point to homogeneous coordinates
        point_homogeneous = np.array([[point[0], point[1]]], dtype=np.float32)

        # Apply perspective transformation
        transformed_point = cv2.perspectiveTransform(point_homogeneous[None, :, :], self.transformation_matrix)

        return (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))
    
    def inverse_transform_point(self, point):
        """
        Transform a point from board coordinates back to frame coordinates
        Args:
            point (tuple): Board space point (x, y)
        Returns:
            tuple: Frame space point (x, y)
        """
        if self.inverse_matrix is None:
            return point

        point_homogeneous = np.array([[point[0], point[1]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point_homogeneous[None, :, :], self.inverse_matrix)

        return (int(transformed_point[0][0][0]), int(transformed_point[0][0][1]))


class SquareMapper:
    """
    Maps detected piece bounding boxes to chess squares (a1-h8) using the
    board's perspective transformation. Uses the bottom-center of each
    bounding box as the piece's ground contact point for accurate assignment.
    """

    def __init__(self, perspective_transformer, white_at_bottom=True):
        """
        Args:
            perspective_transformer: Calibrated PerspectiveTransformer.
            white_at_bottom (bool): True when white's pieces appear at the
                                    bottom of the camera frame (standard view).
        """
        self.pt = perspective_transformer
        self.white_at_bottom = white_at_bottom

    def pixel_to_square(self, x, y):
        """
        Convert a frame pixel coordinate to a chess square label.
        Returns None if the point is outside the board or the transform
        is not yet calibrated.
        """
        if self.pt.transformation_matrix is None or self.pt.board_size is None:
            return None

        tx, ty = self.pt.transform_point((x, y))
        sq = self.pt.board_size / 8

        col = int(tx / sq)
        row = int(ty / sq)

        # Allow pieces at edges with tolerance for board corner detection inaccuracy
        # at the near edge (bottom, closest to camera) which can be off due to
        # perspective distortion. Accept row values down to -3 before clamping.
        if not (-1 <= col <= 8 and -3 <= row <= 8):
            return None

        # Clamp to valid board range [0-7]
        col = max(0, min(7, col))
        row = max(0, min(7, row))

        if self.white_at_bottom:
            file = chr(ord('a') + col)
            rank = 8 - row
        else:
            file = chr(ord('h') - col)
            rank = row + 1

        return f"{file}{rank}"

    def map_board_state(self, tracks, frame_num, frame=None, sam_refiner=None):
        """
        Build a square→piece_type dict for the given frame.

        When `frame` and a live `sam_refiner` (SAM3Refiner) are supplied,
        SAM3 tight mask base-points replace the loose YOLO bbox bottom-centres.
        The mask base-point is the bottom-centre of the actual piece silhouette
        — much closer to the board contact point for tall pieces (king, queen,
        rook).  This reduces square-assignment errors by roughly half a square
        on those pieces.

        When two detections land on the same square, the one whose base sits
        lowest in the frame (largest y) wins — it is physically on the square.

        Args:
            tracks (dict): Output of ChessTracker.get_object_tracks.
            frame_num (int): Frame index.
            frame (np.ndarray, optional): BGR frame for SAM3 refinement.
            sam_refiner (SAM3Refiner, optional): Refiner instance.
        Returns:
            dict: e.g. {'e4': 'white-pawn', 'd7': 'black-queen', ...}
        """
        board_state = {}
        conflicts   = {}   # square → y_base of current winner

        # Collect all pieces from both colours in a single list so SAM3 can
        # cache the image encoding and reuse it across all pieces.
        all_pieces = []   # (color_key, track_id, bbox, piece_type)
        for color in ['white_pieces', 'black_pieces']:
            for track_id, info in tracks[color][frame_num].items():
                all_pieces.append((color, track_id, info['bbox'], info['piece_type']))

        if not all_pieces:
            return board_state

        # ── Base-point computation ────────────────────────────────────────────
        use_sam3 = (frame is not None
                    and sam_refiner is not None
                    and sam_refiner.available)

        if use_sam3:
            bboxes       = [p[2] for p in all_pieces]
            piece_types  = [p[3] for p in all_pieces]
            base_pts     = sam_refiner.get_base_points(frame, bboxes,
                                                       piece_types=piece_types)
        else:
            base_pts  = [(int((p[2][0] + p[2][2]) / 2), int(p[2][3]))
                         for p in all_pieces]

        # ── Square assignment ─────────────────────────────────────────────────
        for (color, track_id, bbox, piece_type), (x_base, y_base) in zip(all_pieces, base_pts):
            square = self.pixel_to_square(x_base, y_base)
            if square is None:
                continue
            if square not in conflicts or y_base > conflicts[square]:
                board_state[square] = piece_type
                conflicts[square]   = y_base

        return board_state

    def draw_square_grid(self, frame):
        """
        Project the 8x8 grid back onto the original frame with square labels.
        Useful for verifying that square mapping is correctly calibrated.
        """
        if self.pt.transformation_matrix is None or self.pt.board_size is None:
            return frame

        sq = self.pt.board_size / 8

        for i in range(9):
            top    = self.pt.inverse_transform_point((i * sq, 0))
            bottom = self.pt.inverse_transform_point((i * sq, 8 * sq))
            cv2.line(frame, top, bottom, (0, 255, 255), 1)

            left  = self.pt.inverse_transform_point((0,      i * sq))
            right = self.pt.inverse_transform_point((8 * sq, i * sq))
            cv2.line(frame, left, right, (0, 255, 255), 1)

        for row in range(8):
            for col in range(8):
                cx = (col + 0.5) * sq
                cy = (row + 0.5) * sq
                fx, fy = self.pt.inverse_transform_point((cx, cy))

                if self.white_at_bottom:
                    label = f"{chr(ord('a') + col)}{8 - row}"
                else:
                    label = f"{chr(ord('h') - col)}{row + 1}"

                cv2.putText(frame, label, (fx - 10, fy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

        return frame

    def draw_piece_assignments(self, frame, board_state):
        """
        Overlay each detected piece's square label at its board position.
        """
        if self.pt.board_size is None:
            return frame

        sq = self.pt.board_size / 8

        for square, piece_type in board_state.items():
            file_idx = ord(square[0]) - ord('a')
            rank_idx = int(square[1])

            if self.white_at_bottom:
                col, row = file_idx, 8 - rank_idx
            else:
                col, row = 7 - file_idx, rank_idx - 1

            cx = (col + 0.5) * sq
            cy = (row + 0.5) * sq
            fx, fy = self.pt.inverse_transform_point((cx, cy))

            text_color = (255, 255, 255) if piece_type.startswith('white') else (30, 30, 30)
            bg_color   = (30, 30, 30)   if piece_type.startswith('white') else (255, 255, 255)

            (tw, th), _ = cv2.getTextSize(square, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (fx - 2, fy - th - 4), (fx + tw + 2, fy + 2), bg_color, -1)
            cv2.putText(frame, square, (fx, fy - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1)

        return frame


class SpeedAndDistanceEstimator:
    def __init__(self, pixels_per_square=50):
        self.pixels_per_square = pixels_per_square
        self.previous_positions = {}

    def calculate_distance(self, point1, point2):
        pixel_distance = np.sqrt(
            (point1[0] - point2[0])**2 +
            (point1[1] - point2[1])**2
        )
        return pixel_distance / self.pixels_per_square

    def calculate_speed(self, distance, time_diff):
        return distance / time_diff if time_diff > 0 else 0

    def estimate_piece_movement(self, tracks, fps):
        piece_movements = {}

        for color in ['white_pieces', 'black_pieces']:
            for frame_num, frame_tracks in enumerate(tracks[color]):
                for track_id, piece_info in frame_tracks.items():
                    current_position = piece_info['position']

                    if track_id not in self.previous_positions:
                        self.previous_positions[track_id] = {
                            'position': current_position,
                            'frame': frame_num
                        }
                        continue

                    prev_pos   = self.previous_positions[track_id]['position']
                    prev_frame = self.previous_positions[track_id]['frame']

                    distance  = self.calculate_distance(prev_pos, current_position)
                    time_diff = frame_num - prev_frame
                    speed     = self.calculate_speed(distance, time_diff)

                    piece_movements.setdefault(track_id, {
                        'total_distance': 0,
                        'max_speed': 0,
                        'piece_type': piece_info.get('piece_type', 'Unknown')
                    })

                    piece_movements[track_id]['total_distance'] += distance
                    piece_movements[track_id]['max_speed'] = max(
                        piece_movements[track_id]['max_speed'], speed
                    )

                    self.previous_positions[track_id] = {
                        'position': current_position,
                        'frame': frame_num
                    }

        return piece_movements

    def analyze_piece_interactions(self, tracks, frame_num):
        piece_interactions = {'white_vs_black': []}

        white_pieces = tracks['white_pieces'][frame_num]
        black_pieces = tracks['black_pieces'][frame_num]

        for white_id, white_piece in white_pieces.items():
            white_pos = white_piece['position']
            for black_id, black_piece in black_pieces.items():
                black_pos = black_piece['position']
                distance  = self.calculate_distance(white_pos, black_pos)

                piece_interactions['white_vs_black'].append({
                    'frame':       frame_num,
                    'white_id':    white_id,
                    'black_id':    black_id,
                    'white_piece': white_piece['piece_type'],
                    'black_piece': black_piece['piece_type'],
                    'distance':    distance
                })

        return piece_interactions


class OcclusionDetector:
    """
    Uses MediaPipe HandLandmarker (Tasks API, v0.10+) to detect when a hand
    is present over the chessboard.

    Returns per-frame (hand_over_board: bool, hand_boxes: list[(x1,y1,x2,y2)])
    so the BSM state machine can freeze tracking during occlusion.
    """

    _MODEL_PATH = os.path.expanduser('~/.mediapipe_hand_landmarker.task')
    _MODEL_URL  = ('https://storage.googleapis.com/mediapipe-models/'
                   'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task')

    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        # Download model file on first use
        if not os.path.exists(self._MODEL_PATH):
            print('Downloading MediaPipe hand landmarker model…')
            urllib.request.urlretrieve(self._MODEL_URL, self._MODEL_PATH)
            print('Hand landmarker model downloaded.')

        base_opts = _mp_tasks.BaseOptions(model_asset_path=self._MODEL_PATH)
        options   = _mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = _mp_vision.HandLandmarker.create_from_options(options)

    def detect(self, frame, board_corners):
        """
        Analyse one BGR frame for hands overlapping the board polygon.

        Args:
            frame:         BGR OpenCV frame.
            board_corners: (4, 2) float array of board corner points in frame space,
                           or None if the board hasn't been detected yet.

        Returns:
            (hand_over_board : bool,
             hand_boxes      : list[(x1,y1,x2,y2)],
             hand_landmarks  : list[list[(x,y)]])   — 21 pixel coords per hand
        """
        if board_corners is None:
            return False, [], []

        h, w    = frame.shape[:2]
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = self._detector.detect(mp_img)

        if not result.hand_landmarks:
            return False, [], []

        # Convex hull of board corners — robust to any corner ordering
        hull = cv2.convexHull(
            board_corners.reshape(-1, 1, 2).astype(np.float32))

        hand_over_board = False
        hand_boxes      = []
        hand_landmarks  = []   # list[list[(px_x, px_y)]] — one per hand

        for hand_lms in result.hand_landmarks:
            xs  = [lm.x * w for lm in hand_lms]
            ys  = [lm.y * h for lm in hand_lms]
            pts = [(int(x), int(y)) for x, y in zip(xs, ys)]

            hand_boxes.append((int(min(xs)), int(min(ys)),
                               int(max(xs)), int(max(ys))))
            hand_landmarks.append(pts)

            # Any landmark inside board polygon → hand over board
            for pt in pts:
                if cv2.pointPolygonTest(hull, (float(pt[0]), float(pt[1])), False) >= 0:
                    hand_over_board = True
                    break

        return hand_over_board, hand_boxes, hand_landmarks

    def close(self):
        self._detector.close()


class SAM3Refiner:
    """
    Uses SAM3's interactive predictor (bbox-prompted) to produce tight
    silhouette masks for each chess piece, replacing the loose YOLO bboxes
    with precise piece outlines.

    Why interactive mode over semantic mode:
      SAM3SemanticPredictor is a detection model — it searches the whole
      image for concept instances regardless of the bbox hint, producing
      masks that bleed over squares.  SAM3Predictor (interactive, the mode
      used when loading via SAM('sam3.pt')) is a promptable segmenter: given
      a tight bbox it segments exactly the object inside it, just like SAM2
      but with SAM3's larger, more accurate backbone.

    All N piece bboxes are batched into one forward pass per frame so the
    image encoder only runs once.  Falls back to bbox bottom-centre on any
    failure.
    """

    _MODEL = 'sam2.1_t.pt'   # SAM 2.1 tiny

    # BGR palettes — warm for white pieces, cool for black pieces
    _WARM = [
        (130, 210, 255), (100, 190, 255), (80,  170, 240),
        (150, 220, 255), (110, 200, 245), (90,  180, 250),
    ]
    _COOL = [
        (200, 120,  60), (210, 100,  50), (190, 140,  70),
        (205, 110,  55), (215, 130,  65), (195, 150,  75),
    ]

    def __init__(self):
        try:
            from ultralytics import SAM as _SAM
            self._sam       = _SAM(self._MODEL)
            self._available = True
            print(f"SAM3Refiner: {self._MODEL} loaded — interactive bbox refinement active")
        except Exception as e:
            print(f"SAM3Refiner: model unavailable ({e}) — falling back to bbox base")
            self._sam       = None
            self._available = False

        # Visualization cache
        self._last_masks       = None   # (N, H, W) float numpy or None
        self._last_piece_types = []

    @property
    def available(self):
        return self._available

    @staticmethod
    def mask_base_point(mask_hw: np.ndarray):
        """Return (x, y) at the bottom-centre of a binary mask (H×W float)."""
        ys, xs = np.where(mask_hw > 0.5)
        if len(ys) == 0:
            return None
        max_y   = ys.max()
        base_xs = xs[ys == max_y]
        return int(base_xs.mean()), int(max_y)

    def get_base_points(self, frame: np.ndarray, bboxes: list,
                        piece_types: list = None) -> list:
        """
        Run SAM3 interactive inference on all bboxes in one batched call.
        Returns one (x, y) base-point per bbox — bottom-centre of the tight
        SAM3 mask, or bbox bottom-centre fallback on any failure.
        """
        fallbacks = [(int((b[0]+b[2])/2), int(b[3])) for b in bboxes]
        self._last_masks       = None
        self._last_piece_types = list(piece_types) if piece_types else []

        if not self._available or not bboxes:
            return fallbacks

        try:
            xyxy   = np.array(bboxes, dtype=np.float32)
            result = self._sam(frame, bboxes=xyxy, verbose=False)[0]

            if result.masks is None or result.masks.data is None:
                return fallbacks

            mask_data = result.masks.data.cpu().numpy()   # (N, H, W) float
            self._last_masks = mask_data

            base_points = []
            for i, mask in enumerate(mask_data):
                bp = self.mask_base_point(mask)
                base_points.append(bp if bp is not None else fallbacks[i])

            # Pad in case SAM3 returned fewer masks than bboxes
            while len(base_points) < len(bboxes):
                base_points.append(fallbacks[len(base_points)])

            return base_points

        except Exception as e:
            print(f"SAM3Refiner.get_base_points error: {e}")
            return fallbacks

    def get_mask_features(self, frame: np.ndarray, bboxes: list,
                          piece_types: list = None) -> list:
        """
        Return shape features (area, height, aspect_ratio) per bbox.
        Used for king/queen disambiguation by silhouette morphology.
        """
        fallback = {'area': 0, 'height': 0, 'aspect_ratio': 1.0}
        if not self._available or not bboxes:
            return [fallback.copy() for _ in bboxes]

        try:
            xyxy   = np.array(bboxes, dtype=np.float32)
            result = self._sam(frame, bboxes=xyxy, verbose=False)[0]

            if result.masks is None or result.masks.data is None:
                return [fallback.copy() for _ in bboxes]

            mask_data = result.masks.data.cpu().numpy()
            features  = []
            for mask in mask_data:
                ys, xs = np.where(mask > 0.5)
                if len(ys) == 0:
                    features.append(fallback.copy())
                else:
                    h = int(ys.max() - ys.min() + 1)
                    w = int(xs.max() - xs.min() + 1)
                    features.append({
                        'area':         int((mask > 0.5).sum()),
                        'height':       h,
                        'aspect_ratio': h / max(w, 1),
                    })

            while len(features) < len(bboxes):
                features.append(fallback.copy())
            return features

        except Exception as e:
            print(f"SAM3Refiner.get_mask_features error: {e}")
            return [fallback.copy() for _ in bboxes]

    def draw_last_masks(self, frame: np.ndarray, alpha: float = 0.38):
        """
        Overlay tight SAM3 masks from the last get_base_points() call.
        White pieces get warm tones; black pieces get cool tones.
        """
        if self._last_masks is None:
            return

        overlay = frame.copy()
        w_idx = b_idx = 0

        for i, mask in enumerate(self._last_masks):
            pt       = self._last_piece_types[i] if i < len(self._last_piece_types) else ''
            is_white = pt.startswith('white')
            if is_white:
                color = self._WARM[w_idx % len(self._WARM)];  w_idx += 1
            else:
                color = self._COOL[b_idx % len(self._COOL)];  b_idx += 1
            overlay[mask > 0.5] = color

        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)

        # Contour pass at full opacity for crisp edges
        for i, mask in enumerate(self._last_masks):
            pt       = self._last_piece_types[i] if i < len(self._last_piece_types) else ''
            is_white = pt.startswith('white')
            color    = self._WARM[0] if is_white else self._COOL[0]
            cnts, _  = cv2.findContours((mask > 0.5).astype(np.uint8),
                                         cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, color, 1, cv2.LINE_AA)


class BoardStateManager:
    """
    Tracks board state across frames, detects moves via stable-state diffing,
    and maintains a chess.Board that reflects the actual game in progress.

    Move detection works in three phases:
      1. Accumulate — collect consecutive frames with the same piece layout.
      2. Commit     — once STABILITY_THRESHOLD stable frames are seen, diff
                      against the last committed state.
      3. Validate   — infer the move from the diff and verify it is legal.
    """

    STABILITY_THRESHOLD = 3  # consecutive matching frames before committing
    MAX_PIECE_DELTA = 6       # max change in piece count to accept as plausible
    SIMILARITY_TOLERANCE = 2  # max differing squares to consider two states the same

    # Occlusion state labels
    S_STABLE   = 'STABLE'    # normal tracking
    S_OCCLUDED = 'OCCLUDED'  # hand over board — frozen
    S_SETTLING = 'SETTLING'  # hand gone — waiting for new stable state

    # Standard chess starting position as a square→piece_type dict.
    # Pre-seeding committed_state with this means the very first diff is
    # against the real game start, so moves that happen before the first
    # stable detection window (e.g. the opening move during hand occlusion)
    # can still be inferred instead of being silently swallowed.
    _STARTING_STATE = {
        'a1': 'white-rook',   'b1': 'white-knight', 'c1': 'white-bishop',
        'd1': 'white-queen',  'e1': 'white-king',   'f1': 'white-bishop',
        'g1': 'white-knight', 'h1': 'white-rook',
        'a2': 'white-pawn',   'b2': 'white-pawn',   'c2': 'white-pawn',
        'd2': 'white-pawn',   'e2': 'white-pawn',   'f2': 'white-pawn',
        'g2': 'white-pawn',   'h2': 'white-pawn',
        'a7': 'black-pawn',   'b7': 'black-pawn',   'c7': 'black-pawn',
        'd7': 'black-pawn',   'e7': 'black-pawn',   'f7': 'black-pawn',
        'g7': 'black-pawn',   'h7': 'black-pawn',
        'a8': 'black-rook',   'b8': 'black-knight', 'c8': 'black-bishop',
        'd8': 'black-queen',  'e8': 'black-king',   'f8': 'black-bishop',
        'g8': 'black-knight', 'h8': 'black-rook',
    }

    def __init__(self, stability_threshold=3, debug=False):
        self.stability_threshold = stability_threshold
        self.debug = debug
        # Pre-seed with the standard starting position so the first move diff
        # is against a known 32-piece board, not an empty state.
        self.committed_state = dict(self._STARTING_STATE)
        self.candidate_state = {}   # current candidate for the next committed state
        self.candidate_count = 0    # consecutive frames matching the candidate
        self.chess_board = chess.Board()
        self.chess_board.chess960 = False   # force standard chess, not Chess960
        self.move_history = []      # list of chess.Move objects in game order
        self.move_methods  = []     # parallel list: how each move was detected
        self.last_move = None
        self.last_commit_frame = None   # frame number of most recent commit
        # Occlusion state machine
        self.occlusion_state = self.S_STABLE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, board_state, frame_num=None, hand_present=False):
        """
        Feed the current frame's board state and hand-presence signal.

        State machine:
          STABLE   → hand arrives   → OCCLUDED  (freeze, snapshot committed state)
          OCCLUDED → hand still here→ OCCLUDED  (stay frozen)
          OCCLUDED → hand leaves    → SETTLING  (reset candidate, start re-stabilising)
          SETTLING → hand returns   → OCCLUDED  (freeze again, keep pre-hand snapshot)
          SETTLING → new stable     → STABLE    (commit move vs pre-hand state)

        Returns a confirmed chess.Move when a new move is detected, else None.
        """
        tag = f"[BSM frame={frame_num}]" if frame_num is not None else "[BSM]"

        # ── Occlusion state transitions ───────────────────────────────────────
        if hand_present:
            if self.occlusion_state == self.S_STABLE:
                self.occlusion_state = self.S_OCCLUDED
                self.candidate_count = 0
                if self.debug:
                    print(f"{tag} [OCCLUDED] Hand detected — board state frozen")
            elif self.occlusion_state == self.S_SETTLING:
                # Hand reappeared before we settled — go back to frozen
                self.occlusion_state = self.S_OCCLUDED
                self.candidate_count = 0
                if self.debug:
                    print(f"{tag} [OCCLUDED] Hand returned during settling")
            # All hand-present cases: freeze tracking
            return None, ''

        if self.occlusion_state == self.S_OCCLUDED:
            # Hand just lifted — start fresh accumulation against pre-hand committed state
            self.occlusion_state = self.S_SETTLING
            self.candidate_state = {}
            self.candidate_count = 0
            if self.debug:
                print(f"{tag} [SETTLING] Hand gone — waiting for stable post-move state")

        # ── Normal stability / commit logic (STABLE and SETTLING) ────────────
        if not self._is_plausible(board_state):
            if self.debug:
                committed_n = len(self.committed_state)
                print(f"{tag} HOLD — pieces={len(board_state)} vs "
                      f"committed={committed_n}, delta too large")
            self.candidate_count = 0
            return None, ''

        diffs_from_candidate = self._count_diffs(board_state, self.candidate_state)
        if diffs_from_candidate <= self.SIMILARITY_TOLERANCE:
            self.candidate_count += 1
        else:
            if self.debug:
                print(f"{tag} New candidate (diffs from old={diffs_from_candidate})")
            self.candidate_state = board_state.copy()
            self.candidate_count = 1

        diffs_from_committed = self._count_diffs(self.candidate_state, self.committed_state)

        if self.debug:
            print(f"{tag} [{self.occlusion_state}] "
                  f"stable={self.candidate_count}/{self.stability_threshold}  "
                  f"diffs_from_committed={diffs_from_committed}  "
                  f"pieces={len(board_state)}")

        # In SETTLING (hand just lifted), even a 1-square diff is likely a real
        # move whose source was already invisible to YOLO in committed_state.
        # Use threshold=0 so any diff triggers an inference attempt; the engine
        # layers will reject noise that doesn't resolve to a legal move.
        _commit_min_diff = (0 if self.occlusion_state == self.S_SETTLING
                            else self.SIMILARITY_TOLERANCE)

        if (self.candidate_count >= self.stability_threshold
                and diffs_from_committed > _commit_min_diff):
            self.last_commit_frame = frame_num
            moves, method = self._infer_move(self.committed_state, self.candidate_state)
            self.committed_state = self.candidate_state.copy()
            self.occlusion_state = self.S_STABLE   # always land in STABLE after commit
            if moves:
                for mv in moves:
                    self.move_history.append(mv)
                    self.move_methods.append(method)
                self.last_move = moves[-1]
                if len(moves) == 1:
                    print(f"Move detected [{method}]: {self.get_last_move_san()} "
                          f"({moves[-1].uci()})  frame={frame_num}  —  {self.get_status()}")
                else:
                    print(f"Moves detected [{method}] ({len(moves)}-step): "
                          f"{self._format_move_list(moves)}  frame={frame_num}"
                          f"  —  {self.get_status()}")
                return moves[-1], method
            else:
                if self.debug:
                    lost   = {sq: p for sq, p in self.committed_state.items()
                              if self.candidate_state.get(sq) != p}
                    gained = {sq: p for sq, p in self.candidate_state.items()
                              if self.committed_state.get(sq) != p}
                    print(f"{tag} State committed but no legal move found. "
                          f"lost={list(lost)} gained={list(gained)}")
                # Update committed to the latest stable detection state.
                # Do NOT reset chess_board — keep the engine-authoritative position
                # so BFS has an accurate starting point and visual fill-in stays correct.
                self.committed_state = self.candidate_state.copy()

        return None, ''

    def force_resync(self, board_state, chess_fen=None, gap_moves_uci=None):
        """
        Force-reset all BSM tracking state to a known baseline.

        Called at recording-gap resume frames to prevent diff-based move
        inference from running across a hard cut in the video (where the
        board state jumped because the player made moves while the recording
        was paused).

        board_state:      Current YOLO-observed board state dict (becomes the
                          new committed_state so future diffs are from here).
        chess_fen:        Optional FEN string to sync chess_board to (from the
                          corrected move log).  When None, chess_board is left
                          untouched.
        gap_moves_uci:    Optional list of UCI move strings that were played
                          during the recording gap.  Injected into move_history
                          BEFORE the chess_board is optionally overwritten with
                          chess_fen, so the history stays consistent.
        """
        # 1. Inject any moves that were played while the recording was paused.
        if gap_moves_uci:
            for uci in gap_moves_uci:
                try:
                    mv = chess.Move.from_uci(uci)
                    self.chess_board.push(mv)
                    self.move_history.append(mv)
                    self.move_methods.append('gap_inject')
                    self.last_move = mv
                except Exception as exc:
                    print(f"[BSM] force_resync: could not inject gap move {uci}: {exc}")

        # 2. Optionally override chess_board with the authoritative FEN.
        #    This corrects any drift that accumulated through the gap.
        if chess_fen is not None:
            try:
                self.chess_board = chess.Board(chess_fen)
            except Exception as exc:
                print(f"[BSM] force_resync: invalid FEN '{chess_fen}': {exc}")

        # 3. Reset detection-layer state so the very next frame is treated as
        #    a fresh baseline rather than diffed against the pre-gap state.
        self.committed_state = board_state.copy()
        self.candidate_state = {}
        self.candidate_count = 0
        self.occlusion_state = self.S_STABLE

        n_gap = len(gap_moves_uci) if gap_moves_uci else 0
        print(f"[BSM] force_resync: committed {len(board_state)} pieces  "
              f"gap_moves={n_gap}  fen={'updated' if chess_fen else 'unchanged'}")

    def get_status(self):
        """Human-readable game status string."""
        b = self.chess_board
        if b.is_checkmate():
            winner = "Black" if b.turn == chess.WHITE else "White"
            return f"Checkmate — {winner} wins"
        if b.is_stalemate():
            return "Stalemate"
        if b.is_insufficient_material():
            return "Draw (insufficient material)"
        if b.is_check():
            side = "White" if b.turn == chess.WHITE else "Black"
            return f"{side} in check!"
        side = "White" if b.turn == chess.WHITE else "Black"
        return f"{side} to move"

    def get_last_move_san(self):
        """Return the last move in Standard Algebraic Notation, or ''."""
        if not self.move_history:
            return ""
        try:
            board_copy = chess.Board()
            for move in self.move_history[:-1]:
                if move in board_copy.legal_moves:
                    board_copy.push(move)
                else:
                    # Turn mismatch — flip and try once (handles one missed move)
                    board_copy.turn = not board_copy.turn
                    if move in board_copy.legal_moves:
                        board_copy.push(move)
                    else:
                        board_copy.turn = not board_copy.turn  # restore
            last = self.move_history[-1]
            if last in board_copy.legal_moves:
                return board_copy.san(last)
            # Try flipped turn for SAN generation
            board_copy.turn = not board_copy.turn
            if last in board_copy.legal_moves:
                return board_copy.san(last)
            return last.uci()
        except Exception:
            # Corrupted move history — fall back to UCI notation
            return self.move_history[-1].uci()

    def draw_game_overlay(self, frame):
        """Status bar: whose turn, last move in SAN, move counter."""
        px = 10
        py = frame.shape[0] - 108
        pw, ph = 260, 98
        radius = 10

        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, radius)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, radius, thickness=1)

        # Turn indicator strip along the top edge of the panel
        is_white_turn = self.chess_board.turn == chess.WHITE
        turn_color = C_WHITE_IND if is_white_turn else C_BLACK_IND
        cv2.rectangle(frame, (px + radius, py), (px + pw - radius, py + 4), turn_color, -1)

        # Move counter badge (top-right of panel)
        move_n = len(self.move_history)
        badge_x = px + pw - 54
        draw_rounded_rect(frame, (badge_x, py + 8), (badge_x + 44, py + 26), C_GREEN, radius=4)

        last_san = self.get_last_move_san()
        status   = self.get_status()
        turn_str = '♔ White' if is_white_turn else '♚ Black'

        def _draw(draw, _):
            draw.text((badge_x + 4, py + 9), f"#{move_n}", font=_font(12, bold=True),
                      fill=bgr_to_rgb(C_PANEL_BG))
            draw.text((px + 12, py + 12), turn_str, font=_font(15, bold=True),
                      fill=bgr_to_rgb(C_TEXT))
            if last_san:
                draw.text((px + 12, py + 36), f"Last move", font=_font(11),
                          fill=bgr_to_rgb(C_MUTED))
                draw.text((px + 12, py + 52), last_san, font=_font(18, bold=True),
                          fill=bgr_to_rgb(C_GOLD))
            else:
                draw.text((px + 12, py + 44), "Game start", font=_font(14),
                          fill=bgr_to_rgb(C_MUTED))
            draw.text((px + 12, py + 78), status, font=_font(12),
                      fill=bgr_to_rgb(C_GREEN if 'check' in status.lower() else C_MUTED))

        pil_draw(frame, _draw)

        # Occlusion state badge — only shown when not STABLE
        if self.occlusion_state != self.S_STABLE:
            self._draw_occlusion_badge(frame)

        return frame

    def _draw_occlusion_badge(self, frame):
        """Small pill badge indicating OCCLUDED or SETTLING state."""
        if self.occlusion_state == self.S_OCCLUDED:
            text  = '✋ Hand detected — tracking paused'
            color = (0, 100, 255)   # orange
        else:
            text  = '↺ Settling…'
            color = (0, 200, 200)   # yellow

        fw = frame.shape[1]
        bw = 260
        bx = fw // 2 - bw // 2
        by = 8
        overlay = frame.copy()
        draw_rounded_rect(overlay, (bx, by), (bx + bw, by + 26), C_PANEL_BG, radius=6)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        draw_rounded_rect(frame, (bx, by), (bx + bw, by + 26), color, radius=6, thickness=2)

        def _draw(draw, _):
            f  = _font(12, bold=True)
            tw = f.getlength(text)
            draw.text(((fw - tw) / 2, by + 5), text, font=f,
                      fill=(*bgr_to_rgb(color), 255))

        pil_draw(frame, _draw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_plausible(self, board_state):
        """Reject states with suspiciously few pieces (hand occluding the board).

        Uses a relative floor (60 % of committed count) rather than a fixed ±6
        delta, because YOLO misses ~35 % of pieces on non-standard sets (e.g.
        Chessnut Evo 3D-printed pieces vs. wooden training data).  The hard
        minimum of 12 catches extreme occlusion (hand covering most of board).
        """
        min_pieces = max(10, int(len(self.committed_state) * 0.55))
        return len(board_state) >= min_pieces

    def _count_diffs(self, state_a, state_b):
        """Count the number of squares that differ between two states."""
        all_squares = set(state_a) | set(state_b)
        return sum(1 for sq in all_squares if state_a.get(sq) != state_b.get(sq))

    def _states_similar(self, state_a, state_b):
        return self._count_diffs(state_a, state_b) <= self.SIMILARITY_TOLERANCE

    def _infer_move(self, prev, curr):
        """
        Infer a legal move sequence from the diff between two committed states.
        Returns (list[chess.Move], method_str) — moves already pushed onto chess_board.
        method_str is one of: 'single', 'castling', 'bfs', 'detection', ''
        """
        # ── Pre-flight Bridge: sync chess_board to 'prev' BSM state ─────────
        # chess_board can drift behind the BSM committed state when a move is
        # played just before a hand occlusion window (YOLO freezes, BSM misses
        # the first move, but commits it in the next stable state as 'prev').
        # Example: Nf3 happens → hand appears → both Nf3 and Nc6 are inside the
        # occluded window.  After the hand lifts BSM committed f3-knight before
        # occlusion, so prev={f3:white-knight} but chess_board still has g1:white-
        # knight (never pushed).  Bridge detects this drift and pushes Nf3 before
        # any inference layer runs — so Layer 1/2/3 see the correct turn.
        _cb_state = self._state_from_chess_board(self.chess_board)
        _bridge_applied: list = []
        if _cb_state:
            # Compute the raw diff between chess_board and prev (before filtering)
            _bridge_all_sq_diff = {sq for sq in set(_cb_state) | set(prev)
                                   if _cb_state.get(sq) != prev.get(sq)}
            _bridge_candidates = self._infer_moves_from_detection(_cb_state, prev)
            # Only fire bridge when diff is CLEAN and small:
            #   • 1-2 candidate moves total
            #   • Total raw diff ≤ 6 squares (allows 1-2 moves + minor YOLO noise)
            # A "clean" single-move diff has exactly 2 changed squares (src + dst,
            # same piece type).  With ≤6 we tolerate up to 2 extra noise squares.
            # This covers: Nf3 missed inside occlusion (diff≈2-4 squares).
            _bridge_clean = (
                1 <= len(_bridge_candidates) <= 2
                and len(_bridge_all_sq_diff) <= 6
            )
            if _bridge_clean:
                for _bm in _bridge_candidates:
                    try:
                        if _bm in self.chess_board.legal_moves:
                            self.chess_board.push(_bm)
                            _bridge_applied.append(_bm)
                    except Exception:
                        pass
                if _bridge_applied and self.debug:
                    print(f"[BSM] Bridge: applied {len(_bridge_applied)} catch-up move(s): "
                          f"{' '.join(m.uci() for m in _bridge_applied)}")

        lost   = {sq: p for sq, p in prev.items() if curr.get(sq) != p}
        gained = {sq: p for sq, p in curr.items() if prev.get(sq) != p}

        # ── Pre-filter: same-square YOLO type-change noise ───────────────────
        # A square in BOTH lost AND gained means YOLO classified the piece
        # differently frame-to-frame (e.g., white-knight → white-bishop on b1).
        # That's a misclassification, not a real move — strip it out.
        same_sq = set(lost) & set(gained)
        if same_sq:
            lost   = {sq: p for sq, p in lost.items()   if sq not in same_sq}
            gained = {sq: p for sq, p in gained.items() if sq not in same_sq}

        if not lost and not gained:
            # prev == curr after noise filtering — bridge may have caught up
            if _bridge_applied:
                return _bridge_applied, 'bridge'
            return [], ''

        # ── Layer 1: Single-move inference (engine-validated) ─────────────────

        # Castling — python-chess encodes it as a king move only
        if len(lost) == 2 and len(gained) == 2:
            for move in self.chess_board.legal_moves:
                if self.chess_board.is_castling(move):
                    fs = chess.square_name(move.from_square)
                    ts = chess.square_name(move.to_square)
                    if fs in lost and ts in gained:
                        self.chess_board.push(move)
                        return _bridge_applied + [move], 'castling'

        # Simple move or capture
        for from_sq in lost:
            for to_sq in gained:
                move = self._try_push(from_sq, to_sq, lost[from_sq])
                if move:
                    return _bridge_applied + [move], 'single'

        # Fallback: scan all legal moves for the best square match
        for move in self.chess_board.legal_moves:
            fs = chess.square_name(move.from_square)
            ts = chess.square_name(move.to_square)
            if fs in lost and ts in gained:
                self.chess_board.push(move)
                return _bridge_applied + [move], 'single'

        # ── Layer 1.5: "Appeared piece" inference ──────────────────────────────
        # The piece's source square was already absent from committed_state (YOLO
        # missed it), so `lost` is empty.  Use the engine to find which legal move
        # lands the right piece type on every gained square.
        if not lost and gained and len(gained) <= 2:
            for to_sq, piece_type in list(gained.items()):
                color_str, ptype_str = piece_type.split('-')
                for move in self.chess_board.legal_moves:
                    if chess.square_name(move.to_square) != to_sq:
                        continue
                    moved_piece = self.chess_board.piece_at(move.from_square)
                    if moved_piece is None:
                        continue
                    if (self._COLOR_NAME.get(moved_piece.color) == color_str and
                            self._PTYPE_NAME.get(moved_piece.piece_type) == ptype_str):
                        if self.debug:
                            print(f"[BSM] Layer1.5: appeared piece on {to_sq} → "
                                  f"{move.uci()} (source was YOLO-invisible)")
                        self.chess_board.push(move)
                        return _bridge_applied + [move], 'single'

        # ── Layer 2: BFS from prev detection state ────────────────────────────
        if self.debug:
            print(f"[BSM] Single-move failed (lost={list(lost.keys())} "
                  f"gained={list(gained.keys())}) — BFS depth 2–5 from prev_state")
        seq = self._bfs_move_sequence(prev, curr, max_depth=5)
        if seq:
            if self.debug:
                print(f"[BSM] BFS → {len(seq)}-step: "
                      f"{' '.join(mv.uci() for mv in seq)}")
            # Apply BFS moves on a working copy so we never partially mutate
            # chess_board.  The pre-flight bridge above already synced chess_board
            # to 'prev', so every move in seq must be legal on chess_board or we
            # discard it — no YOLO-board fallback (that was the old corrupt path).
            work = self.chess_board.copy()
            applied = []
            for mv in seq:
                if mv not in work.legal_moves:
                    # BFS found a move that's illegal on the authoritative board.
                    # The YOLO diff is noisy — stop here rather than resetting to
                    # a corrupt YOLO-derived board.
                    break
                work.push(mv)
                applied.append(mv)
            if applied:
                self.chess_board = work
                return _bridge_applied + applied, 'bfs'

        if self.debug:
            print("[BSM] BFS: no legal sequence found within depth 5")

        # ── Layer 3: Detection-based (YOLO diff, no engine validation) ────────
        # Pass chess_board so illegal candidates (e.g. blocked rook path from a1→h1
        # when pieces newly appear in YOLO) are filtered out before push attempts.
        det_moves = self._infer_moves_from_detection(prev, curr, board=self.chess_board)
        if det_moves:
            if self.debug:
                print(f"[BSM] Detection-based → {len(det_moves)} move(s): "
                      f"{' '.join(mv.uci() for mv in det_moves)}")
            # Push moves that are legal; track which ones were actually pushed.
            # Use a retry loop: after pushing one move the turn changes, which
            # can make a previously-skipped move (e.g. Nc6 after Nf3) legal.
            det_pushed = []
            remaining = list(det_moves)
            while remaining:
                made_progress = False
                for mv in list(remaining):
                    try:
                        if mv in self.chess_board.legal_moves:
                            self.chess_board.push(mv)
                            det_pushed.append(mv)
                            remaining.remove(mv)
                            made_progress = True
                            break  # restart — turn just changed, re-check all remaining
                    except Exception:
                        remaining.remove(mv)  # remove invalid move to avoid infinite loop
                if not made_progress:
                    break  # no remaining move is legal; stop

            # Turn-flip recovery: if nothing pushed, chess_board's turn is off by
            # one half-move due to a missed intermediate move.  The detection
            # filter in _infer_moves_from_detection already verified each move is
            # legal on the flipped turn, so applying the flip here is safe.
            if not det_pushed and remaining:
                self.chess_board.turn = not self.chess_board.turn
                if self.debug:
                    side = 'white' if self.chess_board.turn == chess.WHITE else 'black'
                    print(f"[BSM] Layer3: normal push failed — flipping turn to {side} "
                          f"and retrying {len(remaining)} move(s)")
                retry_remaining = list(remaining)
                while retry_remaining:
                    made_progress = False
                    for mv in list(retry_remaining):
                        try:
                            if mv in self.chess_board.legal_moves:
                                self.chess_board.push(mv)
                                det_pushed.append(mv)
                                retry_remaining.remove(mv)
                                made_progress = True
                                break
                        except Exception:
                            retry_remaining.remove(mv)
                    if not made_progress:
                        break
                if not det_pushed:
                    # Flip didn't help — restore the original turn
                    self.chess_board.turn = not self.chess_board.turn
                    if self.debug:
                        print("[BSM] Layer3: turn-flip also failed — restoring turn")

            # Only prepend bridge moves if they ENABLED some det_move to be pushed.
            # If no det_moves were pushed (wrong turn, etc.), exclude bridge moves
            # from move_history — otherwise a spurious bridge move corrupts SAN
            # computation for the unpushed det_moves that follow.
            if det_pushed:
                return _bridge_applied + det_pushed, 'detection'
            else:
                # All layers failed — return empty so the commit still advances
                # committed_state but nothing is added to move_history.
                return [], ''

        # No move found in any layer — return bridge-only moves if we have them
        if _bridge_applied:
            return _bridge_applied, 'bridge'
        return [], ''

    def _infer_moves_from_detection(self, prev, curr, board=None):
        """
        Infer moves purely from YOLO position diffs — no chess engine needed.

        Strategy:
          1. Find pieces of the same (color, type) that disappeared from one
             square and appeared on another.
          2. Apply a quick geometry filter (rook can't jump diagonally, etc.).
          3. For captures: a piece disappears from sq_A, a same-color piece
             appears at sq_A's old square (the opponent's piece is simply gone).

        board: optional chess.Board — when provided, candidates are filtered
               against board.legal_moves so blocked/illegal moves are dropped
               before being returned.  Pass None when called from the bridge
               (which intentionally searches without turn constraints).

        Returns list[chess.Move] (UCI format, may include promotion 'q').
        No moves are pushed here — caller is responsible.
        """
        lost   = {sq: p for sq, p in prev.items() if curr.get(sq) != p}
        gained = {sq: p for sq, p in curr.items() if prev.get(sq) != p}

        if not lost and not gained:
            return []

        # Filter same-square type-change noise (YOLO misclassification)
        same_sq = set(lost) & set(gained)
        if same_sq:
            lost   = {sq: p for sq, p in lost.items()   if sq not in same_sq}
            gained = {sq: p for sq, p in gained.items() if sq not in same_sq}

        if not lost and not gained:
            return []

        # Reject if change is implausibly large after noise filtering
        # (12 = up to 4 moves missed + capture + noise margin)
        if len(lost) + len(gained) > 12:
            return []

        moves    = []
        used_from = set()
        used_to   = set()

        # Pass 1 — straightforward: same piece type disappeared + appeared
        for from_sq, piece_type in sorted(lost.items()):
            for to_sq, gained_piece in sorted(gained.items()):
                if to_sq in used_to or from_sq in used_from:
                    continue
                if piece_type == gained_piece:   # same color+type moved
                    if self._piece_move_geometry_ok(from_sq, to_sq, piece_type):
                        try:
                            uci  = f"{from_sq}{to_sq}"
                            parts = piece_type.split('-')
                            # Auto-promote pawns reaching back rank
                            if parts[-1] == 'pawn':
                                dest_rank = int(to_sq[1])
                                if (parts[0] == 'white' and dest_rank == 8) or \
                                   (parts[0] == 'black' and dest_rank == 1):
                                    uci += 'q'
                            moves.append(chess.Move.from_uci(uci))
                            used_from.add(from_sq)
                            used_to.add(to_sq)
                            break
                        except Exception:
                            pass

        # Pass 2 — captures: piece moved to a square where opponent piece was
        for from_sq, piece_type in sorted(lost.items()):
            if from_sq in used_from:
                continue
            color = piece_type.split('-')[0]
            for to_sq in sorted(gained.keys()):
                if to_sq in used_to:
                    continue
                # to_sq had an opponent piece in prev and our piece is there now
                prev_at_dest = prev.get(to_sq, '')
                curr_at_dest = curr.get(to_sq, '')
                opp = 'black' if color == 'white' else 'white'
                if prev_at_dest.startswith(opp) and curr_at_dest == piece_type:
                    if self._piece_move_geometry_ok(from_sq, to_sq, piece_type):
                        try:
                            moves.append(chess.Move.from_uci(f"{from_sq}{to_sq}"))
                            used_from.add(from_sq)
                            used_to.add(to_sq)
                            break
                        except Exception:
                            pass

        # Sliding-piece path filter: for rooks, bishops, and queens, verify
        # that the path between from- and to-squares was clear in the source
        # state (prev).  This blocks spurious matches like a1→h1 when the
        # path through b1-g1 is occupied — a common false positive when YOLO
        # starts detecting pieces it previously missed (e.g. fine-tuned model).
        # Non-sliding pieces (pawns, knights, kings) are unaffected.
        # When board is provided, also apply a full legality filter for truly
        # illegal moves, but PRESERVE wrong-turn moves (legal on other side) —
        # those are real game moves played during an occlusion window.
        if board is not None:
            filtered = []
            for mv in moves:
                if mv in board.legal_moves:
                    filtered.append(mv)
                    continue
                # Check if it would be legal if we flip the turn
                # (catches real moves played in the wrong YOLO-turn order)
                board_flip = board.copy()
                board_flip.turn = not board_flip.turn
                if mv in board_flip.legal_moves:
                    filtered.append(mv)  # real move, wrong turn order — keep
                # else: truly illegal (path blocked, etc.) — discard
            moves = filtered

        return moves

    @staticmethod
    def _piece_move_geometry_ok(from_sq, to_sq, piece_type):
        """Quick sanity check: does this move match the piece's movement rules?"""
        try:
            f  = chess.parse_square(from_sq)
            t  = chess.parse_square(to_sq)
            fr, ff = divmod(f, 8)
            tr, tf = divmod(t, 8)
            dr, df = abs(tr - fr), abs(tf - ff)
            ptype  = piece_type.split('-')[-1]
            if ptype == 'pawn':
                return df <= 1 and 1 <= dr <= 2
            if ptype == 'knight':
                return sorted([dr, df]) == [1, 2]
            if ptype == 'bishop':
                return dr == df and dr > 0
            if ptype == 'rook':
                return (dr == 0 or df == 0) and (dr + df > 0)
            if ptype == 'queen':
                return (dr == df > 0) or ((dr == 0 or df == 0) and dr + df > 0)
            if ptype == 'king':
                return max(dr, df) <= 2 and (dr + df > 0)   # ≤2 allows castling
        except Exception:
            pass
        return True   # unknown piece — allow it

    def _bfs_move_sequence(self, prev_state, target_state, max_depth=5):
        """
        BFS from a board built from prev_state (the last stable detection) to
        reach target_state.  Starting from the detection state — rather than
        self.chess_board — means BFS only needs to cover the 1–3 moves between
        consecutive detected positions, regardless of how far self.chess_board
        has drifted behind the real game.

        Only explores branches touching squares that changed between prev and
        target (aggressive pruning).

        Returns list[chess.Move] (moves NOT yet pushed) or [].
        """
        from collections import deque

        # Build starting board from prev detection state
        start = self._build_board_from_state(prev_state, self.chess_board.turn)
        if start is None:
            if self.debug:
                print("[BSM] BFS: prev_state missing a king — skipping")
            return []

        # Squares that changed between prev and target
        changed = {sq for sq in set(prev_state) | set(target_state)
                   if prev_state.get(sq) != target_state.get(sq)}
        if not changed:
            return []

        queue   = deque([(start, [])])
        visited = {start.board_fen()}

        while queue:
            board, moves = queue.popleft()
            if len(moves) >= max_depth:
                continue
            for move in board.legal_moves:
                fs = chess.square_name(move.from_square)
                ts = chess.square_name(move.to_square)
                # Prune: only follow moves that touch a changed square
                if fs not in changed and ts not in changed:
                    continue
                nb = board.copy()
                nb.push(move)
                fk = nb.board_fen()
                if fk in visited:
                    continue
                visited.add(fk)
                new_moves = moves + [move]
                if self._board_matches_state(nb, target_state):
                    return new_moves
                queue.append((nb, new_moves))
        return []

    @staticmethod
    def _board_to_state(board):
        """Convert a python-chess Board to our {square_name: 'color-type'} dict."""
        state = {}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                color_str = 'white' if piece.color == chess.WHITE else 'black'
                type_str  = chess.piece_name(piece.piece_type)
                state[chess.square_name(sq)] = f"{color_str}-{type_str}"
        return state

    def _board_matches_state(self, board, target_state):
        """True if board's layout matches target_state within SIMILARITY_TOLERANCE."""
        return self._count_diffs(self._board_to_state(board), target_state) \
               <= self.SIMILARITY_TOLERANCE

    _PTYPE_MAP = {
        'pawn': chess.PAWN, 'knight': chess.KNIGHT, 'bishop': chess.BISHOP,
        'rook': chess.ROOK, 'queen': chess.QUEEN, 'king': chess.KING,
    }

    def _build_board_from_state(self, state, turn=None):
        """
        Build a chess.Board from our detection state dict.
        Sets castling rights based on king/rook positions (conservative).
        Returns None if either king is missing (invalid position).
        """
        board = chess.Board(fen=None)
        board.clear_board()
        for sq_name, piece_str in state.items():
            try:
                sq    = chess.parse_square(sq_name)
                parts = piece_str.split('-')
                color = chess.WHITE if parts[0] == 'white' else chess.BLACK
                ptype = self._PTYPE_MAP.get(parts[-1], chess.PAWN)
                board.set_piece_at(sq, chess.Piece(ptype, color))
            except Exception:
                continue
        # Both kings must be present for a valid board
        if not board.king(chess.WHITE) or not board.king(chess.BLACK):
            return None
        board.turn = turn if turn is not None else chess.WHITE
        # Conservative castling rights: only if king+rook are on original squares
        board.castling_rights = chess.BB_EMPTY
        if board.piece_at(chess.E1) == chess.Piece(chess.KING, chess.WHITE):
            if board.piece_at(chess.H1) == chess.Piece(chess.ROOK, chess.WHITE):
                board.castling_rights |= chess.BB_H1
            if board.piece_at(chess.A1) == chess.Piece(chess.ROOK, chess.WHITE):
                board.castling_rights |= chess.BB_A1
        if board.piece_at(chess.E8) == chess.Piece(chess.KING, chess.BLACK):
            if board.piece_at(chess.H8) == chess.Piece(chess.ROOK, chess.BLACK):
                board.castling_rights |= chess.BB_H8
            if board.piece_at(chess.A8) == chess.Piece(chess.ROOK, chess.BLACK):
                board.castling_rights |= chess.BB_A8
        return board

    _COLOR_NAME = {chess.WHITE: 'white', chess.BLACK: 'black'}
    _PTYPE_NAME = {
        chess.PAWN:   'pawn',   chess.KNIGHT: 'knight', chess.BISHOP: 'bishop',
        chess.ROOK:   'rook',   chess.QUEEN:  'queen',  chess.KING:   'king',
    }

    def _state_from_chess_board(self, board):
        """
        Derive our detection-state dict (sq_name → 'color-type') from a chess.Board.
        Returns None if the board is invalid (missing king).
        """
        if not board.is_valid():
            return None
        state = {}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is not None:
                color_str = self._COLOR_NAME[piece.color]
                ptype_str = self._PTYPE_NAME[piece.piece_type]
                state[chess.square_name(sq)] = f"{color_str}-{ptype_str}"
        return state

    def _format_move_list(self, moves):
        """Return a SAN string list for a sequence of moves already in move_history."""
        # Replay from the board position N moves ago to get SAN strings
        board = self.chess_board.copy()
        for _ in moves:
            board.pop()
        parts = []
        for mv in moves:
            parts.append(board.san(mv))
            board.push(mv)
        return '  '.join(parts)

    def _try_push(self, from_sq, to_sq, piece_type):
        """Try pushing a move (with automatic queen-promotion for pawns)."""
        try:
            move = chess.Move.from_uci(f"{from_sq}{to_sq}")
            if move in self.chess_board.legal_moves:
                self.chess_board.push(move)
                return move

            if 'pawn' in piece_type:
                promo = chess.Move.from_uci(f"{from_sq}{to_sq}q")
                if promo in self.chess_board.legal_moves:
                    self.chess_board.push(promo)
                    return promo
        except ValueError:
            pass
        return None


# ── ECO opening table ─────────────────────────────────────────────────────────

_ECO_CACHE_PATH = os.path.expanduser('~/.chess_eco_cache.pkl')
_ECO_BASE_URL   = ('https://raw.githubusercontent.com/'
                   'lichess-org/chess-openings/master/{}.tsv')


def _build_eco_table() -> dict:
    """
    Download the lichess ECO opening database (a–e TSV files) and return a
    dict mapping board EPD → (eco_code, opening_name).  Results are cached
    locally so the download only happens once.
    """
    if os.path.exists(_ECO_CACHE_PATH):
        try:
            with open(_ECO_CACHE_PATH, 'rb') as f:
                table = pickle.load(f)
            print(f"ECO table loaded from cache: {len(table)} openings")
            return table
        except Exception:
            pass  # corrupted cache → rebuild

    table = {}
    for letter in 'abcde':
        url = _ECO_BASE_URL.format(letter)
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                lines = resp.read().decode('utf-8').splitlines()
            for line in lines[1:]:          # skip header row
                parts = line.split('\t')
                if len(parts) < 3:
                    continue
                eco, name, pgn_text = parts[0], parts[1], parts[2]
                try:
                    game = chess.pgn.read_game(io.StringIO(pgn_text))
                    board = chess.Board()
                    for move in game.mainline_moves():
                        board.push(move)
                    table[board.epd()] = (eco, name)
                except Exception:
                    pass
        except Exception as e:
            print(f"Warning: could not fetch ECO data for '{letter}': {e}")

    try:
        with open(_ECO_CACHE_PATH, 'wb') as f:
            pickle.dump(table, f)
    except Exception:
        pass
    print(f"ECO table built: {len(table)} openings, cached to {_ECO_CACHE_PATH}")
    return table


class GameStateEngine:
    """
    Analysis and presentation layer that sits on top of BoardStateManager.

    Responsibilities:
      - PGN generation and export
      - Stockfish evaluation (optional — skipped gracefully if unavailable)
      - Evaluation bar overlay
      - Move history panel (last 8 full moves, highlighted current)
      - Last-move square highlighting projected back onto the camera frame
    """

    def __init__(self, board_state_manager, stockfish_path=None, fps=30.0):
        self.bsm = board_state_manager
        self._fps = max(1.0, fps)   # frames-per-second for move-time calculation
        self.engine = None
        self._san_cache = []
        self._san_cache_len = 0
        self.last_eval = None
        self._eval_at_move = -1
        self.captured_by_white = []   # piece types (chess.BLACK pieces white captured)
        self.captured_by_black = []   # piece types (chess.WHITE pieces black captured)
        # Effective capture lists — YOLO-inferred when detection data is rich enough,
        # otherwise falls back to move-log.  These drive the material bar and
        # captured-pieces panel so they stay accurate even when moves are missed.
        self._eff_cap_white = []
        self._eff_cap_black = []
        self._move_annotations = []   # (symbol, bgr_color) per move in move_history
        self._best_move             = None  # engine's top suggestion for the current position
        self._engine_preferred_move = None  # what engine would have played instead of last move
        self._ghost_arrow_ttl       = 0     # frames remaining to show ghost arrow
        self._capture_flash_square  = None  # square to flash on a capture
        self._capture_flash_ttl     = 0     # frames remaining for capture flash
        self._last_eval_depth       = 0     # depth used in the most recent Stockfish analysis
        # Analysis board — rebuilt from YOLO committed_state so Stockfish sees
        # the actual piece positions rather than only the confirmed-move log.
        self._analysis_board       = chess.Board()   # starts at standard position
        self._committed_state_key  = None            # hash of last synced state
        # Eval history for the graph — one entry per committed move
        self._eval_history     = []   # list[int | None]  centipawn values
        self._eval_graph_move  = -1  # move_count at which we last appended
        # Latest detected board state — cached so draw helpers can read it
        self._last_detected_state = {}
        # Per-move CPL history (None when Stockfish was unavailable)
        self._cpl_history          = []
        # Move-time tracking (seconds per committed move)
        self._move_times           = []
        self._last_move_frame      = 0
        # Critical moment flash overlay
        self._critical_flash_ttl   = 0
        self._critical_flash_label = ""
        self._critical_move_indices = set()   # move indices flagged as turning points
        # Frame counter for pulsing animations
        self._frame_count = 0
        self._opening_eco  = ""
        self._opening_name = ""
        try:
            self._eco_table = _build_eco_table()
        except Exception as e:
            print(f"Warning: ECO table unavailable — {e}")
            self._eco_table = {}

        self._stockfish_path = stockfish_path   # kept so we can restart after a crash
        if stockfish_path:
            try:
                self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                print(f"Stockfish loaded: {stockfish_path}")
            except Exception as e:
                print(f"Warning: Stockfish unavailable — {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def notify_move(self, move, frame_num=0):
        """Call whenever BoardStateManager confirms a new move.

        Args:
            move:      the chess.Move that was committed
            frame_num: current video frame (used to compute move duration)
        """
        # Move-time: frames elapsed since last committed move → seconds
        elapsed_s = (frame_num - self._last_move_frame) / self._fps
        self._move_times.append(round(elapsed_s, 1))
        self._last_move_frame = frame_num

        # Classify move quality via Stockfish: temporarily undo the move to
        # get the pre-move eval, then restore and compare.
        annotation = ('!', C_ANNOT_GOOD)
        cpl        = None
        if self.engine:
            was_white = (self.bsm.chess_board.turn == chess.BLACK)
            pre_cp = None
            # Only pop if the move on top of the stack is actually THIS move.
            # Detection-based moves may not have been pushed to chess_board
            # (because they were illegal on chess_board's current turn), so
            # blindly popping would corrupt the board state.
            move_is_on_stack = (
                bool(self.bsm.chess_board.move_stack)
                and self.bsm.chess_board.move_stack[-1] == move
            )
            try:
                if not move_is_on_stack:
                    raise ValueError("move not on chess_board stack — skip Stockfish pop/push")
                self.bsm.chess_board.pop()
                # Detect capture before re-pushing (board is in pre-move state)
                if self.bsm.chess_board.is_capture(move):
                    self._capture_flash_square = move.to_square
                    self._capture_flash_ttl    = 20   # ~0.8 s at 25 fps
                # Validate board is legal before sending to Stockfish
                if not self.bsm.chess_board.is_valid():
                    raise ValueError("board invalid after pop — skip Stockfish")
                info = self.engine.analyse(
                    self.bsm.chess_board, chess.engine.Limit(depth=14, time=8.0))
                s = info['score'].white()
                pre_cp = (10000 if (s.is_mate() and s.mate() > 0)
                          else -10000 if s.is_mate()
                          else s.score())
                # Capture what the engine preferred over the move that was played
                pv = info.get('pv', [])
                self._engine_preferred_move = pv[0] if pv else None
            except Exception:
                pre_cp = None
                self._engine_preferred_move = None
            finally:
                if move_is_on_stack:
                    try:
                        self.bsm.chess_board.push(move)
                    except Exception:
                        pass   # board was reset mid-analysis; skip re-push

            if pre_cp is not None:
                self._eval_at_move = -1          # force post-move re-analysis
                post_cp = self._get_evaluation(depth=14)
                if post_cp is not None:
                    sf = 1 if was_white else -1
                    # CPL = centipawn loss for the mover (positive = bad)
                    cpl = sf * (pre_cp - post_cp)
                    annotation = self._classify_move_quality(cpl)

                    # Critical-moment detection: flag and trigger flash overlay.
                    # Thresholds must match _classify_move_quality bands:
                    #   ?? blunder  > 600 CPL  → "💥 Blunder!" banner
                    #   ?  mistake  > 300 CPL  → "⚡ Critical Moment" banner
                    #   ?! inaccuracy > 150 CPL → no banner (minor annotation only)
                    move_idx = len(self._cpl_history)   # index before appending
                    if cpl > 600:
                        self._critical_move_indices.add(move_idx)
                        self._critical_flash_label = "💥  Blunder!"
                        self._critical_flash_ttl   = 90
                    elif cpl > 300:
                        self._critical_move_indices.add(move_idx)
                        self._critical_flash_label = "⚡  Critical Moment"
                        self._critical_flash_ttl   = 75

        self._cpl_history.append(cpl)

        # Ghost arrow visible for 50 frames (~2 s) after each move
        self._ghost_arrow_ttl = 50

        self._move_annotations.append(annotation)
        self._rebuild_san_cache()
        self._recompute_captures()
        self._update_opening()

    def annotate_frame(self, frame, square_mapper, detected_board_state=None):
        """
        Apply all game-engine annotations in draw order:
          1. last-move square highlight (behind everything)
          2. move arrow on top of highlight
          3. evaluation bar (right edge)
          4. move history panel (top-right)
          5. captured pieces + material badge (below stats panel)

        Args:
            detected_board_state: optional {square: piece_type} dict from the
                current frame's YOLO detections — used to drive the distribution
                panel so it reflects the live board, not just confirmed moves.
        """
        self._frame_count += 1

        # Cache the live detection dict so draw helpers (e.g. _draw_ghost_arrow)
        # can validate square occupancy without needing extra parameters.
        self._last_detected_state = detected_board_state or {}

        # Sync material balance from YOLO detections (runs every frame so the
        # capture counts and material bar stay accurate even when moves are missed)
        self._sync_effective_captures(self._last_detected_state)
        self._sync_analysis_board(self._last_detected_state)

        # Board control heatmap — deepest layer (everything draws on top)
        frame = self._draw_board_control_heatmap(frame, square_mapper)

        # Capture flash (behind everything else)
        if self._capture_flash_ttl > 0:
            frame = self._draw_capture_flash(frame, square_mapper)
            self._capture_flash_ttl -= 1

        frame = self._draw_check_and_threats(frame, square_mapper)
        frame = self._draw_last_move_highlight(frame, square_mapper)

        # Ghost arrow — only for the TTL window after the move
        if self._ghost_arrow_ttl > 0:
            frame = self._draw_ghost_arrow(frame, square_mapper)
            self._ghost_arrow_ttl -= 1

        frame = self._draw_move_arrow(frame, square_mapper)
        frame = self._draw_move_quality_badge(frame, square_mapper)
        frame = self._draw_board_coordinates(frame, square_mapper)
        frame = self._draw_hanging_pieces(frame, square_mapper)
        frame = self._draw_eval_bar(frame, self._get_evaluation())
        frame = self._draw_move_history(frame)
        frame = self.draw_captured_pieces(frame)
        frame = self._draw_opening_name(frame)
        frame = self._draw_material_bar(frame)
        frame = self._draw_king_safety_panel(frame)
        frame = self._draw_accuracy_panel(frame)
        frame = self._draw_engine_panel(frame)
        frame = self._draw_eval_graph(frame)
        frame = self._draw_piece_distribution(frame, detected_board_state)
        # Critical-moment flash banner (drawn last so it sits on top)
        if self._critical_flash_ttl > 0:
            frame = self._draw_critical_flash(frame)
            self._critical_flash_ttl -= 1
        return frame

    def get_pgn(self):
        """Return the full game as a PGN string."""
        game = chess.pgn.Game()
        game.headers["Event"] = "Live Detection"
        game.headers["Date"] = "????.??.??"
        node = game
        board = chess.Board()
        for move in self.bsm.move_history:
            node = node.add_variation(move)
            board.push(move)
        buf = io.StringIO()
        print(game, file=buf)
        return buf.getvalue()

    def save_pgn(self, path):
        """Write the current game to a PGN file."""
        with open(path, 'w') as f:
            f.write(self.get_pgn())
        print(f"PGN saved to: {path}")

    def close(self):
        """Shut down the Stockfish engine cleanly."""
        if self.engine:
            try:
                self.engine.quit()
            except Exception:
                pass   # engine may have already died; that's fine
            self.engine = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _rebuild_san_cache(self):
        """Incrementally extend the SAN cache when new moves arrive."""
        target = len(self.bsm.move_history)
        if target == self._san_cache_len:
            return
        board = chess.Board()
        try:
            for move in self.bsm.move_history[:self._san_cache_len]:
                try:
                    if move in board.legal_moves:
                        board.push(move)
                except Exception:
                    pass
            for move in self.bsm.move_history[self._san_cache_len:]:
                try:
                    if move in board.legal_moves:
                        self._san_cache.append(board.san(move))
                        board.push(move)
                    else:
                        self._san_cache.append(move.uci())
                except Exception:
                    self._san_cache.append(move.uci())
            self._san_cache_len = target
            return  # success — skip the except truncation
        except Exception as exc:
            # A move in history is illegal from the initial position — this can
            # happen when BFS returned moves relative to a drifted detection
            # state rather than the true game sequence.  Truncate move_history
            # to the last verified-legal prefix so the game display stays sane.
            print(f"[GameEngine] _rebuild_san_cache error ({exc}); "
                  f"chess960={self.bsm.chess_board.chess960} "
                  f"history={[m.uci() for m in self.bsm.move_history]} truncating...", flush=True)
            board2 = chess.Board()
            valid = 0
            for mv in self.bsm.move_history:
                if mv not in board2.legal_moves:
                    break
                board2.push(mv)
                valid += 1
            self.bsm.move_history  = self.bsm.move_history[:valid]
            self.bsm.move_methods  = self.bsm.move_methods[:valid]
            self._san_cache        = self._san_cache[:valid]
            self._san_cache_len    = valid

    def _get_evaluation(self, depth=12):
        """Return cached centipawn score from white's POV, or None.
        Also caches the engine's top suggestion in self._best_move for the ghost arrow."""
        if self.engine is None:
            return None
        move_count = len(self.bsm.move_history)
        if move_count == self._eval_at_move:
            return self.last_eval
        try:
            # Skip analysis if board is in an illegal state (e.g. after detection reset)
            if not self._analysis_board.is_valid():
                return self.last_eval
            info = self.engine.analyse(
                self._analysis_board,
                chess.engine.Limit(depth=depth)
            )
            score = info["score"].white()
            if score.is_mate():
                self.last_eval = 10000 if score.mate() > 0 else -10000
            else:
                self.last_eval = score.score()
            # Cache engine's top suggestion for ghost arrow overlay
            pv = info.get("pv", [])
            self._best_move       = pv[0] if pv else None
            self._last_eval_depth = depth
            self._eval_at_move    = move_count
            # Accumulate eval history for the graph (one slot per committed move;
            # update the current slot if the YOLO position refreshed mid-move)
            if move_count > self._eval_graph_move:
                self._eval_history.append(self.last_eval)
                self._eval_graph_move = move_count
            elif move_count == self._eval_graph_move and self._eval_history:
                self._eval_history[-1] = self.last_eval   # position updated, refresh slot
        except Exception as e:
            print(f"Stockfish error: {e}")
            # Engine may have died (EngineTerminatedError); try to restart it once
            try:
                self.engine.close()
            except Exception:
                pass
            self.engine = None
            if self._stockfish_path:
                try:
                    self.engine = chess.engine.SimpleEngine.popen_uci(self._stockfish_path)
                    print("[Stockfish] engine restarted after crash")
                    self._eval_at_move = -1   # retry evaluation next frame
                except Exception as restart_err:
                    print(f"[Stockfish] restart failed: {restart_err}")
        return self.last_eval

    def _classify_move_quality(self, cpl_cp):
        """Map centipawn loss (positive = bad for the mover) to an annotation.
        Thresholds calibrated for depth=14 Stockfish; bands loosened ~50%
        vs. Lichess to absorb the extra eval noise at shallower depth.

        CPL bands:
          !!  ≤ -30  : outperforms engine's top line (noise-filtered)
          !   ≤ 150  : solid move, within ~1.5 pawns of best
          ?!  ≤ 300  : inaccuracy, 1.5–3 pawns below best
          ?   ≤ 600  : mistake, 3–6 pawns below best
          ??  > 600  : blunder, 6+ pawns below best
        """
        if cpl_cp <= -30:
            return ('!!', C_ANNOT_BRILLIANT)   # clearly better than engine's line
        elif cpl_cp <= 150:
            return ('!',  C_ANNOT_GOOD)        # good move, within ~1.5 pawns
        elif cpl_cp <= 300:
            return ('?!', C_ANNOT_INACCUR)     # inaccuracy, 1.5–3 pawns
        elif cpl_cp <= 600:
            return ('?',  C_ANNOT_MISTAKE)     # mistake, 3–6 pawns
        else:
            return ('??', C_ANNOT_BLUNDER)     # blunder, 6+ pawns

    def _square_corners(self, chess_sq, square_mapper):
        """
        Return the 4 frame-space corners of a chess square as an int32 array.
        Handles both white_at_bottom orientations.
        """
        sq_px = square_mapper.pt.board_size / 8
        fi = chess.square_file(chess_sq)
        ri = chess.square_rank(chess_sq)

        if square_mapper.white_at_bottom:
            col, row = fi, 7 - ri
        else:
            col, row = 7 - fi, ri

        corners_board = [
            (col * sq_px,       row * sq_px),
            ((col + 1) * sq_px, row * sq_px),
            ((col + 1) * sq_px, (row + 1) * sq_px),
            (col * sq_px,       (row + 1) * sq_px),
        ]
        return np.array(
            [square_mapper.pt.inverse_transform_point(p) for p in corners_board],
            dtype=np.int32
        )

    def _draw_board_control_heatmap(self, frame, square_mapper):
        """
        Project a board-control heatmap onto the camera frame.

        For each of the 64 squares, `chess.Board.attackers()` is called for
        both sides and the resulting square polygon (mapped back to camera
        space via inverse_transform_point) is tinted:

          Amber/gold  — white attacks only
          Steel-blue  — black attacks only
          Purple      — contested (both sides attack)

        Unattacked squares receive no tint.  The overlay is blended at 25 %
        opacity so the board texture, pieces and other annotations remain clear.

        Uses the confirmed chess.Board state so the heatmap reflects legal
        attack patterns for the current position (not raw YOLO detections).
        """
        if square_mapper.pt.board_size is None:
            return frame

        board = self.bsm.chess_board

        # BGR colours — chosen to complement the existing chess.com palette
        _C_WHITE   = ( 60, 165, 230)   # amber / gold  (white side)
        _C_BLACK   = (185,  65,  55)   # steel blue    (black side)
        _C_CONTEST = (175,  65, 175)   # purple        (contested)

        # Build three separate overlay images (one per tint category) so we
        # can blend each at a consistent alpha in just three addWeighted calls
        # instead of one per square.
        ov_w = frame.copy()   # white-only attack squares
        ov_b = frame.copy()   # black-only attack squares
        ov_c = frame.copy()   # contested squares
        drew_w = drew_b = drew_c = False

        for sq in chess.SQUARES:
            w_att = bool(board.attackers(chess.WHITE, sq))
            b_att = bool(board.attackers(chess.BLACK, sq))

            if not w_att and not b_att:
                continue

            corners = self._square_corners(sq, square_mapper)

            if w_att and b_att:
                cv2.fillPoly(ov_c, [corners], _C_CONTEST)
                drew_c = True
            elif w_att:
                cv2.fillPoly(ov_w, [corners], _C_WHITE)
                drew_w = True
            else:
                cv2.fillPoly(ov_b, [corners], _C_BLACK)
                drew_b = True

        alpha = 0.25
        if drew_w:
            cv2.addWeighted(ov_w, alpha, frame, 1.0 - alpha, 0, frame)
        if drew_b:
            cv2.addWeighted(ov_b, alpha, frame, 1.0 - alpha, 0, frame)
        if drew_c:
            cv2.addWeighted(ov_c, alpha + 0.03, frame, 1.0 - alpha - 0.03, 0, frame)

        return frame

    def _draw_check_and_threats(self, frame, square_mapper):
        """
        Overlay two layers of positional warnings:
          • Threatened pieces: squares where the side-to-move has a piece that
            is attacked by the opponent — soft orange tint (20 % opacity).
          • Check: the king square of the side in check — bold red tint (55 %).
        Both layers are drawn before move arrows so the arrow remains on top.
        """
        board = self.bsm.chess_board
        if square_mapper.pt.board_size is None:
            return frame

        side      = board.turn
        opponent  = not side
        overlay   = frame.copy()
        drew_any  = False

        # ── Threatened pieces (hanging / en prise) ──────────────────────────
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.color != side:
                continue
            if board.attackers(opponent, sq):
                corners = self._square_corners(sq, square_mapper)
                cv2.fillPoly(overlay, [corners], (0, 100, 255))   # orange-red
                drew_any = True

        if drew_any:
            cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
            overlay = frame.copy()

        # ── Check highlight ──────────────────────────────────────────────────
        if board.is_check():
            king_sq = board.king(side)
            if king_sq is not None:
                corners = self._square_corners(king_sq, square_mapper)
                cv2.fillPoly(overlay, [corners], (0, 0, 220))     # vivid red
                cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
                overlay = frame.copy()

                # Pulsing border ring around the king square
                cv2.polylines(frame, [corners], True, (0, 0, 255), 3, cv2.LINE_AA)

                # "CHECK" badge centred on the king square
                cx = int(corners[:, 0].mean())
                cy = int(corners[:, 1].mean())

                def _badge(draw, _):
                    txt = "CHECK"
                    f   = _font(13, bold=True)
                    bb  = draw.textbbox((0, 0), txt, font=f)
                    tw, th = bb[2] - bb[0], bb[3] - bb[1]
                    bx, by = cx - tw // 2 - 5, cy - th // 2 - 3
                    draw.rounded_rectangle(
                        [bx, by, bx + tw + 10, by + th + 6],
                        radius=4, fill=(220, 0, 0)
                    )
                    draw.text((bx + 5, by + 3), txt, font=f,
                              fill=(255, 255, 255))

                pil_draw(frame, _badge)

        return frame

    def _update_opening(self):
        """Check the current board position against the ECO table and keep the last match."""
        if not self._eco_table:
            return
        epd = self.bsm.chess_board.epd()
        if epd in self._eco_table:
            self._opening_eco, self._opening_name = self._eco_table[epd]

    def _draw_opening_name(self, frame):
        """Render an opening-name badge below the captured-pieces panel."""
        if not self._opening_name:
            return frame

        # Wrap long names: split on ': ' or ',' near the middle
        name = self._opening_name
        eco  = self._opening_eco
        px, py = 10, 200
        pw     = 230
        radius = 10

        # Split on ': ' for variant names, e.g. "Sicilian Defense: Najdorf Variation"
        if ': ' in name:
            base, variant = name.split(': ', 1)
        elif len(name) > 22:
            # Break at last space before char 22
            idx = name.rfind(' ', 0, 22)
            base, variant = (name[:idx], name[idx+1:]) if idx != -1 else (name, '')
        else:
            base, variant = name, ''

        line2_h = 14 if variant else 0
        ph = 20 + 16 + line2_h + 10   # header + eco+base + [variant] + padding

        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, radius)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        # Green left accent
        cv2.rectangle(frame, (px, py + radius), (px + 3, py + ph - radius), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, radius, thickness=1)

        def _draw(draw, _):
            draw.text((px + 10, py + 4), "Opening", font=_font(11, bold=True),
                      fill=bgr_to_rgb(C_MUTED))
            # ECO badge + opening base name on same row
            draw.text((px + 10, py + 19), eco, font=_font(12, bold=True),
                      fill=bgr_to_rgb(C_GREEN))
            draw.text((px + 10 + 32, py + 19), base, font=_font(12),
                      fill=bgr_to_rgb(C_TEXT))
            if variant:
                draw.text((px + 10, py + 34), variant, font=_font(11),
                          fill=bgr_to_rgb(C_MUTED))

        pil_draw(frame, _draw)
        return frame

    def _draw_material_bar(self, frame):
        """Horizontal material-balance panel below the opening badge.

        Layout (top → bottom inside the panel):
          • Header row: "Material"  +  net-score badge
          • Balance bar: thin horizontal bar, center-anchored, white fills right / black fills left
          • Piece-diff rows: icons showing which types each side is ahead on
        """
        px, py = 10, 265
        pw, ph = 230, 88
        radius = 10

        # ── Compute piece-type imbalances ────────────────────────────────────
        # captured_by_white = black pieces white took  →  white is "up" these types
        # captured_by_black = white pieces black took  →  black is "up" these types
        PIECE_ORDER = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        BLACK_SYM   = {chess.QUEEN: '♛', chess.ROOK: '♜', chess.BISHOP: '♝',
                       chess.KNIGHT: '♞', chess.PAWN: '♟'}
        WHITE_SYM   = {chess.QUEEN: '♕', chess.ROOK: '♖', chess.BISHOP: '♗',
                       chess.KNIGHT: '♘', chess.PAWN: '♙'}

        w_score, b_score = self._material_advantage()
        net = w_score - b_score  # positive = White ahead

        # Build icon strings: white_up (black piece icons) / black_up (white piece icons)
        white_up_str = ''
        black_up_str = ''
        for pt in PIECE_ORDER:
            w = self._eff_cap_white.count(pt)
            b = self._eff_cap_black.count(pt)
            diff = w - b
            if diff > 0:
                white_up_str += BLACK_SYM[pt] * diff   # White captured these black pieces
            elif diff < 0:
                black_up_str += WHITE_SYM[pt] * (-diff) # Black captured these white pieces

        # ── Panel background ─────────────────────────────────────────────────
        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, radius)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px, py + radius), (px + 3, py + ph - radius), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, radius, thickness=1)

        # ── Balance bar ──────────────────────────────────────────────────────
        bar_y  = py + 32
        bar_h  = 8
        bar_x  = px + 10
        bar_w  = pw - 20
        cx     = bar_x + bar_w // 2          # centre of bar
        max_cp = 39                           # full material = 39 pts per side

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (50, 47, 44), -1)

        if net != 0:
            fill_px = min(bar_w // 2, int(abs(net) / max_cp * (bar_w // 2)))
            if net > 0:   # White ahead — fill right of centre
                cv2.rectangle(frame, (cx, bar_y), (cx + fill_px, bar_y + bar_h), C_EVAL_WHITE, -1)
            else:          # Black ahead — fill left of centre
                cv2.rectangle(frame, (cx - fill_px, bar_y), (cx, bar_y + bar_h), C_EVAL_BLACK, -1)

        # Centre tick
        cv2.rectangle(frame, (cx - 1, bar_y - 2), (cx + 1, bar_y + bar_h + 2), C_PANEL_BORDER, -1)

        # ── Text ─────────────────────────────────────────────────────────────
        if net > 0:
            score_str, score_col = f"+{net}", C_EVAL_WHITE
        elif net < 0:
            score_str, score_col = str(net), (180, 165, 150)
        else:
            score_str, score_col = "=", C_MUTED

        def _draw(draw, _):
            draw.text((px + 10, py + 6), "Material",
                      font=_font(11, bold=True), fill=bgr_to_rgb(C_MUTED))
            # Net score badge, right-aligned
            draw.text((px + pw - 32, py + 6), score_str,
                      font=_font(13, bold=True), fill=bgr_to_rgb(score_col))
            # Piece diff icons — two rows below the bar
            draw.text((px + 10, bar_y + bar_h + 8),
                      white_up_str or '—', font=_font(13),
                      fill=bgr_to_rgb(C_CREAM))
            draw.text((px + 10, bar_y + bar_h + 26),
                      black_up_str or '—', font=_font(13),
                      fill=bgr_to_rgb(C_MUTED))

        pil_draw(frame, _draw)
        return frame

    def _recompute_captures(self):
        """Replay the full move history on a scratch board to collect captured pieces."""
        cap_w, cap_b = [], []
        board = chess.Board()
        for move in self.bsm.move_history:
            captured = board.piece_at(move.to_square)
            if board.is_en_passant(move):
                # pawn captured via en passant — always a pawn
                if board.turn == chess.WHITE:
                    cap_w.append(chess.PAWN)
                else:
                    cap_b.append(chess.PAWN)
            elif captured is not None:
                if board.turn == chess.WHITE:
                    cap_w.append(captured.piece_type)
                else:
                    cap_b.append(captured.piece_type)
            board.push(move)
        self.captured_by_white = cap_w
        self.captured_by_black = cap_b

    def _sync_effective_captures(self, detected_board_state):
        """
        Derive capture lists from YOLO-detected piece counts vs the standard
        starting position.  Pieces absent from the board are presumed captured.

        Uses YOLO data when ≥ 10 pieces are detected (enough to be reliable).
        Falls back to the move-log (`captured_by_white/black`) otherwise — e.g.
        during heavy occlusion or in the very first frames.

        Updates self._eff_cap_white and self._eff_cap_black in-place.
        """
        if not detected_board_state or len(detected_board_state) < 10:
            self._eff_cap_white = list(self.captured_by_white)
            self._eff_cap_black = list(self.captured_by_black)
            return

        # Tally detected pieces by colour and type
        on_board = {chess.WHITE: {}, chess.BLACK: {}}
        for pt_str in detected_board_state.values():
            parts = pt_str.split('-')
            if len(parts) != 2:
                continue
            color_str, type_str = parts
            color = chess.WHITE if color_str == 'white' else chess.BLACK
            pt    = _PT_FROM_STR.get(type_str)
            if pt is not None:
                on_board[color][pt] = on_board[color].get(pt, 0) + 1

        # Missing pieces (vs standard start) are inferred as captured
        cap_by_white = []   # black pieces gone → white captured them
        cap_by_black = []   # white pieces gone → black captured them

        for pt, start in _PIECE_START.items():
            if pt == chess.KING:
                continue   # kings are never captured in legal chess
            b_missing = max(0, start - on_board[chess.BLACK].get(pt, 0))
            cap_by_white.extend([pt] * b_missing)
            w_missing = max(0, start - on_board[chess.WHITE].get(pt, 0))
            cap_by_black.extend([pt] * w_missing)

        self._eff_cap_white = cap_by_white
        self._eff_cap_black = cap_by_black

    def _sync_analysis_board(self, detected_board_state):
        """Rebuild _analysis_board from YOLO committed_state so Stockfish sees
        actual piece positions rather than only the confirmed-move log.

        Priority order for source data:
          1. bsm.committed_state  — stable multi-frame YOLO consensus (≥10 pieces)
          2. detected_board_state — current-frame fallback (≥10 pieces)
          3. bsm.chess_board      — move-log fallback when detections are sparse
        """
        state_to_use = None
        if len(self.bsm.committed_state) >= 10:
            state_to_use = self.bsm.committed_state
        elif detected_board_state and len(detected_board_state) >= 10:
            state_to_use = detected_board_state

        if not state_to_use:
            self._analysis_board = self.bsm.chess_board
            return

        state_key = hash(frozenset(state_to_use.items()))
        if state_key == self._committed_state_key:
            return   # nothing changed — skip expensive FEN rebuild

        self._committed_state_key = state_key
        self._eval_at_move = -1   # force Stockfish re-evaluation with new position

        ref = self.bsm.chess_board   # borrow turn / move number from move log
        try:
            rows = []
            for rank in range(8, 0, -1):
                empty, row = 0, ''
                for f in 'abcdefgh':
                    pt = state_to_use.get(f'{f}{rank}')
                    fc = _PIECE_FEN_CHAR.get(pt, '') if pt else ''
                    if fc:
                        if empty:
                            row += str(empty)
                            empty = 0
                        row += fc
                    else:
                        empty += 1
                if empty:
                    row += str(empty)
                rows.append(row)
            turn = 'w' if ref.turn == chess.WHITE else 'b'
            fen  = f"{'/'.join(rows)} {turn} - - 0 {ref.fullmove_number}"
            candidate = chess.Board(fen)
            # Reject positions that would confuse or crash Stockfish:
            #   missing kings, pawns on back ranks, opponent in check, etc.
            if (candidate.king(chess.WHITE) is None or
                    candidate.king(chess.BLACK) is None):
                raise ValueError("YOLO state missing a king")
            if not candidate.is_valid():
                raise ValueError(
                    "YOLO state produces illegal position "
                    "(pawn on back rank / opponent in check / other) — using move-log board"
                )
            self._analysis_board = candidate
        except Exception as e:
            print(f"[sync_analysis_board] FEN build failed: {e}")
            self._analysis_board = self.bsm.chess_board

    def _material_advantage(self):
        """Return (white_score, black_score) from effective (YOLO-inferred) captures."""
        w = sum(PIECE_VALUES.get(pt, 0) for pt in self._eff_cap_white)
        b = sum(PIECE_VALUES.get(pt, 0) for pt in self._eff_cap_black)
        return w, b

    def _square_center(self, chess_sq, square_mapper):
        """Return the frame-space (x, y) center of a chess square."""
        sq_px = square_mapper.pt.board_size / 8
        fi = chess.square_file(chess_sq)
        ri = chess.square_rank(chess_sq)
        if square_mapper.white_at_bottom:
            col, row = fi, 7 - ri
        else:
            col, row = 7 - fi, ri
        bx = (col + 0.5) * sq_px
        by = (row + 0.5) * sq_px
        return square_mapper.pt.inverse_transform_point((bx, by))

    def _draw_move_quality_badge(self, frame, square_mapper):
        """Render the move-quality symbol (e.g. '!', '??') at the destination square."""
        if not self._move_annotations or self.bsm.last_move is None:
            return frame
        if square_mapper.pt.board_size is None:
            return frame

        sym, color = self._move_annotations[-1]
        cx, cy = (int(v) for v in self._square_center(
            self.bsm.last_move.to_square, square_mapper))

        sq_px = int(square_mapper.pt.board_size / 8)
        r     = max(10, sq_px // 5)       # badge radius scales with square size

        # Filled circle background
        cv2.circle(frame, (cx + sq_px // 4, cy - sq_px // 4), r, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx + sq_px // 4, cy - sq_px // 4), r, color,        2,  cv2.LINE_AA)

        def _badge(draw, _):
            bx = cx + sq_px // 4 - r + 2
            by = cy - sq_px // 4 - r + 2
            f  = _font(max(9, r - 2), bold=True)
            draw.text((bx, by), sym, font=f, fill=(*bgr_to_rgb(color), 255))

        pil_draw(frame, _badge)
        return frame

    def _draw_capture_flash(self, frame, square_mapper):
        """Brief red/orange flash on the square where a capture just occurred."""
        if self._capture_flash_square is None or square_mapper.pt.board_size is None:
            return frame
        corners = self._square_corners(self._capture_flash_square, square_mapper)
        # Intensity fades linearly from 1.0 → 0.0 over the TTL window
        alpha = self._capture_flash_ttl / 20.0
        overlay = frame.copy()
        cv2.fillPoly(overlay, [corners], (30, 80, 220))   # red-orange in BGR
        cv2.addWeighted(overlay, alpha * 0.55, frame, 1 - alpha * 0.55, 0, frame)
        return frame

    def _draw_engine_panel(self, frame):
        """Engine analysis panel anchored to the bottom-right.
        Shows current eval, search depth, and Stockfish's best-move suggestion."""
        if self.engine is None or self.last_eval is None:
            return frame

        # Eval label
        ev = self.last_eval
        if abs(ev) >= 10000:
            eval_str  = "M+" if ev > 0 else "M-"
            eval_col  = C_EVAL_WHITE if ev > 0 else C_EVAL_BLACK
        else:
            eval_str  = f"{ev / 100:+.2f}"
            eval_col  = C_EVAL_WHITE if ev >= 0 else (180, 160, 140)

        # Best-move label (SAN of engine's top suggestion for the current position)
        best_str = "—"
        if self._best_move is not None:
            try:
                best_str = self.bsm.chess_board.san(self._best_move)
            except Exception:
                best_str = self._best_move.uci()

        depth_str = str(self._last_eval_depth) if self._last_eval_depth else "—"

        pw, ph   = 210, 90
        radius   = 10
        fh, fw   = frame.shape[:2]
        px = fw - pw - 38
        py = fh - 178   # bottom of stack: eval-graph(72) + gap(6) + engine(90) + margin(10)

        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, radius)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px + radius, py), (px + pw - radius, py + 3), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, radius, thickness=1)

        # Divider between header and rows
        div_y = py + 26
        cv2.line(frame, (px + 10, div_y), (px + pw - 10, div_y), C_PANEL_BORDER, 1)

        def _draw(draw, _):
            # Header
            draw.text((px + 10, py + 6), "Engine Analysis",
                      font=_font(13, bold=True), fill=bgr_to_rgb(C_MUTED))

            # Row 1 — Eval (large) + Depth (right-aligned label)
            draw.text((px + 10, div_y + 6),
                      "Eval", font=_font(11), fill=bgr_to_rgb(C_MUTED))
            draw.text((px + 48, div_y + 4),
                      eval_str, font=_font(15, bold=True), fill=bgr_to_rgb(eval_col))
            draw.text((px + 118, div_y + 6),
                      "Depth", font=_font(11), fill=bgr_to_rgb(C_MUTED))
            draw.text((px + 162, div_y + 6),
                      depth_str, font=_font(13, bold=True), fill=bgr_to_rgb(C_TEXT))

            # Row 2 — Best move
            draw.text((px + 10, div_y + 30),
                      "Best", font=_font(11), fill=bgr_to_rgb(C_MUTED))
            draw.text((px + 48, div_y + 28),
                      best_str, font=_font(15, bold=True), fill=bgr_to_rgb(C_GREEN))

        pil_draw(frame, _draw)
        return frame

    def _draw_eval_graph(self, frame):
        """Filled-area evaluation history graph, anchored below the engine panel.

        Layout (same width as engine panel, 72 px tall):
          • Dark panel with rounded corners + green top-accent
          • Thin "Eval History" label
          • Chart area: white fill above zero, dark fill below, green polyline on top
          • Zero gridline across the middle; rightmost point marked with a dot
        Y-axis clamped to ±500 cp (≈ ±5 pawns); mate pinned to the edge.
        """
        if not self._eval_history:
            return frame

        pw, gh   = 210, 72
        radius   = 10
        fh, fw   = frame.shape[:2]
        px       = fw - pw - 38
        py       = fh - gh - 10      # 10 px bottom margin

        # --- Panel background ---
        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + gh), C_PANEL_BG, radius)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px + radius, py), (px + pw - radius, py + 3), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + gh), C_PANEL_BORDER, radius, thickness=1)

        # --- Header label ---
        def _hdr(draw, _):
            draw.text((px + 10, py + 6), "Eval History",
                      font=_font(11, bold=True), fill=bgr_to_rgb(C_MUTED))
        pil_draw(frame, _hdr)

        # --- Chart geometry ---
        hdr_h  = 20          # pixels reserved for the header
        pad    = 6
        cx0    = px + pad
        cy0    = py + hdr_h
        cw     = pw - 2 * pad
        ch     = gh - hdr_h - pad
        zero_y = cy0 + ch // 2

        # Zero gridline
        cv2.line(frame, (cx0, zero_y), (cx0 + cw, zero_y), C_PANEL_BORDER, 1)

        # Clamp & normalise eval to pixel y-coordinate
        def _ev_y(ev):
            if ev is None:
                return zero_y
            c = max(-500, min(500, ev))
            # +500 cp → top of chart (cy0), -500 cp → bottom (cy0+ch)
            return int(zero_y - c * (ch / 2) / 500)

        # Subsample if more history points than chart pixels
        hist = self._eval_history
        n    = len(hist)
        if n > cw:
            step  = n / cw
            hist  = [hist[int(i * step)] for i in range(cw)]
            n     = len(hist)

        if n < 1:
            return frame

        xs = [int(cx0 + i * (cw - 1) / max(n - 1, 1)) for i in range(n)]
        ys = [_ev_y(v) for v in hist]
        pts = list(zip(xs, ys))

        # White fill (above zero) — polygon: left-zero → curve (clamped ≤ zero_y) → right-zero
        w_curve = [(x, min(y, zero_y)) for x, y in pts]
        w_poly  = np.array(
            [(cx0, zero_y)] + w_curve + [(xs[-1], zero_y)],
            dtype=np.int32)
        cv2.fillPoly(frame, [w_poly], C_EVAL_WHITE)

        # Black fill (below zero) — curve (clamped ≥ zero_y)
        b_curve = [(x, max(y, zero_y)) for x, y in pts]
        b_poly  = np.array(
            [(cx0, zero_y)] + b_curve + [(xs[-1], zero_y)],
            dtype=np.int32)
        cv2.fillPoly(frame, [b_poly], C_EVAL_BLACK)

        # Green evaluation polyline on top
        cv2.polylines(frame,
                      [np.array(pts, dtype=np.int32)],
                      isClosed=False,
                      color=C_GREEN,
                      thickness=1,
                      lineType=cv2.LINE_AA)

        # Critical-moment markers — small downward triangle below the polyline
        for mi in self._critical_move_indices:
            if 0 <= mi < n:
                mx = xs[mi] if mi < len(xs) else xs[-1]
                my = ys[mi] if mi < len(ys) else ys[-1]
                tip_y  = min(cy0 + ch - 2, my + 6)
                tri = np.array([
                    [mx,     my - 1],
                    [mx - 4, my - 7],
                    [mx + 4, my - 7],
                ], dtype=np.int32)
                cv2.fillPoly(frame, [tri], C_ANNOT_BLUNDER)

        # Current-eval dot (rightmost point)
        cv2.circle(frame, pts[-1], 3, C_GREEN, -1, cv2.LINE_AA)
        cv2.circle(frame, pts[-1], 3, C_TEXT,   1, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    # Accuracy panel
    # ------------------------------------------------------------------

    @staticmethod
    def _cpl_to_accuracy(cpl):
        """Chess.com-style accuracy % from centipawn loss (equal-position baseline)."""
        if cpl is None:
            return None
        cpl = max(0, cpl)
        wp_before = 0.5
        wp_after  = 1.0 / (1.0 + 10.0 ** (cpl / 400.0))
        win_pct_diff = (wp_before - wp_after) * 100.0
        return max(0.0, min(100.0,
                            103.1668 * math.exp(-0.04354 * win_pct_diff) - 3.1669))

    def _player_accuracy(self, color):
        """Return average accuracy % for the given color, or None if no data."""
        start = 0 if color == chess.WHITE else 1
        accs  = [self._cpl_to_accuracy(self._cpl_history[i])
                 for i in range(start, len(self._cpl_history), 2)
                 if self._cpl_history[i] is not None]
        return round(sum(accs) / len(accs), 1) if accs else None

    def _draw_accuracy_panel(self, frame):
        """Two-row panel showing running accuracy % for white and black."""
        pw, ph = 210, 55
        fh, fw = frame.shape[:2]
        px     = fw - pw - 38
        py     = fh - 239   # above engine(90+gap6) above eval-graph(72+margin10)

        w_acc = self._player_accuracy(chess.WHITE)
        b_acc = self._player_accuracy(chess.BLACK)

        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, 10)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px + 10, py), (px + pw - 10, py + 3), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, 10, thickness=1)

        bar_w  = 100   # max bar width in pixels
        bar_h  = 7
        gap    = 6

        def _acc_row(draw, label, acc, col_label, row_y, bar_col):
            # Label
            draw.text((px + 10, row_y), label, font=_font(11), fill=bgr_to_rgb(C_MUTED))
            # Percentage
            pct_str = f"{acc:.0f}%" if acc is not None else "—"
            draw.text((px + 46, row_y - 1), pct_str,
                      font=_font(13, bold=True), fill=bgr_to_rgb(col_label))
            # Bar (drawn via cv2 below)

        def _draw(draw, _):
            draw.text((px + 10, py + 5), "Accuracy",
                      font=_font(11, bold=True), fill=bgr_to_rgb(C_MUTED))
            _acc_row(draw, "W", w_acc, C_EVAL_WHITE, py + 21, C_EVAL_WHITE)
            _acc_row(draw, "B", b_acc, C_TEXT,        py + 38, C_TEXT)

        pil_draw(frame, _draw)

        # Draw filled accuracy bars via cv2 (after pil_draw so they're on top)
        for row_off, acc, bar_col in [(21, w_acc, C_EVAL_WHITE),
                                       (38, b_acc, (140, 135, 128))]:
            bx    = px + 86
            by    = py + row_off + 3
            track_x2 = bx + bar_w
            cv2.rectangle(frame, (bx, by), (track_x2, by + bar_h),
                          C_PANEL_BORDER, -1)
            if acc is not None:
                fill_w = int(bar_w * acc / 100)
                cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h),
                              bar_col, -1)

        return frame

    # ------------------------------------------------------------------
    # King safety panel
    # ------------------------------------------------------------------

    def _king_safety_score(self, board, color):
        """Return 0-100 safety score (100 = very safe)."""
        king_sq = board.king(color)
        if king_sq is None:
            return 50
        king_file = chess.square_file(king_sq)
        king_rank = chess.square_rank(king_sq)
        opponent  = not color

        # 1. Pawn shield: own pawns one rank ahead of king
        pawn_dir    = 1 if color == chess.WHITE else -1
        shield_rank = king_rank + pawn_dir
        shield      = 0
        if 0 <= shield_rank <= 7:
            for df in (-1, 0, 1):
                f = king_file + df
                if 0 <= f <= 7:
                    p = board.piece_at(chess.square(f, shield_rank))
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        shield += 1

        # 2. Attackers on squares adjacent to king
        attack_count = len(board.attackers(opponent, king_sq))
        for adj_sq in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
            attack_count += len(board.attackers(opponent, adj_sq))

        # 3. Open files adjacent to king (own pawn absent → easier rook/queen lift)
        open_files = 0
        for df in (-1, 0, 1):
            f = king_file + df
            if not (0 <= f <= 7):
                continue
            has_pawn = any(
                board.piece_at(chess.square(f, r)) == chess.Piece(chess.PAWN, color)
                for r in range(8)
            )
            if not has_pawn:
                open_files += 1

        score = 100 - open_files * 10 - max(0, 3 - shield) * 12 - attack_count * 6
        return max(0, min(100, score))

    def _draw_king_safety_panel(self, frame):
        """Compact panel with two king-safety gauges (white and black)."""
        pw, ph  = 210, 50
        fh, fw  = frame.shape[:2]
        px      = fw - pw - 38
        py      = fh - 295   # above accuracy(55+gap6) above engine(90+gap6) above eval-graph(72+margin10)

        board = self._analysis_board

        w_score = self._king_safety_score(board, chess.WHITE)
        b_score = self._king_safety_score(board, chess.BLACK)

        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, 10)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px + 10, py), (px + pw - 10, py + 3), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, 10, thickness=1)

        def _score_col(s):
            """Green → yellow → red based on safety score."""
            if s >= 70: return (60, 180, 75)     # green
            if s >= 40: return (30, 165, 255)    # amber
            return (50, 50, 220)                  # red

        def _draw(draw, _):
            draw.text((px + 10, py + 5), "King Safety",
                      font=_font(11, bold=True), fill=bgr_to_rgb(C_MUTED))
            for label, score, row_y in [("♔", w_score, py + 20),
                                         ("♚", b_score, py + 34)]:
                col = _score_col(score)
                draw.text((px + 10, row_y), label, font=_font(12), fill=bgr_to_rgb(C_TEXT))
                draw.text((px + 30, row_y), f"{score}",
                          font=_font(12, bold=True), fill=bgr_to_rgb(col))

        pil_draw(frame, _draw)

        # Gauge bars
        bar_w = 120; bar_h = 6
        for row_off, score in [(20, w_score), (34, b_score)]:
            bx = px + 58
            by = py + row_off + 3
            cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), C_PANEL_BORDER, -1)
            fill_w = int(bar_w * score / 100)
            col = _score_col(score)
            cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), col, -1)

        return frame

    # ------------------------------------------------------------------
    # Hanging-piece highlight
    # ------------------------------------------------------------------

    def _draw_hanging_pieces(self, frame, square_mapper):
        """Draw pulsing orange border on squares with hanging (undefended attacked) pieces."""
        if square_mapper.pt.board_size is None:
            return frame
        board = self._analysis_board
        if board is None:
            return frame

        # Pulsing alpha: oscillates between 0.25 and 0.55 at ~1 Hz
        pulse = 0.40 + 0.15 * math.sin(self._frame_count * 0.18)

        hanging = []
        _CP = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 99}
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece is None or piece.piece_type == chess.KING:
                continue
            opp = not piece.color
            if not board.is_attacked_by(opp, sq):
                continue
            defenders = board.attackers(piece.color, sq)
            if not defenders:
                hanging.append(sq)
                continue
            # Also flag if cheapest attacker < piece value (winning exchange)
            attackers_sq = board.attackers(opp, sq)
            min_atk = min(
                (_CP.get(board.piece_at(a).piece_type, 0) for a in attackers_sq
                 if board.piece_at(a)),
                default=99)
            if min_atk < _CP.get(piece.piece_type, 0):
                hanging.append(sq)

        if not hanging:
            return frame

        overlay = frame.copy()
        _HANG_COL = (0, 100, 255)   # orange-red in BGR
        for sq in hanging:
            corners = self._square_corners(sq, square_mapper)
            cv2.polylines(overlay, [corners], isClosed=True,
                          color=_HANG_COL, thickness=3, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, pulse, frame, 1.0 - pulse, 0, frame)
        return frame

    # ------------------------------------------------------------------
    # Critical-moment flash banner
    # ------------------------------------------------------------------

    def _draw_critical_flash(self, frame):
        """Fading banner at the top of the frame for blunders / critical moments."""
        if not self._critical_flash_label:
            return frame
        fh, fw = frame.shape[:2]

        # Fade: full opacity for first 30 frames, linear fade for remaining TTL
        max_ttl   = 90
        alpha_raw = self._critical_flash_ttl / max_ttl
        alpha     = min(1.0, alpha_raw * 2)   # ramp up fast, fade slowly

        bh = 38
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, fh // 2 - bh // 2),
                      (fw, fh // 2 + bh // 2), C_PANEL_BG, -1)
        cv2.addWeighted(overlay, 0.78 * alpha, frame, 1.0 - 0.78 * alpha, 0, frame)

        label_col = C_ANNOT_BLUNDER if "Blunder" in self._critical_flash_label else C_GOLD

        def _draw(draw, _):
            f  = _font(18, bold=True)
            tw = f.getlength(self._critical_flash_label)
            draw.text(((fw - tw) / 2, fh / 2 - 12),
                      self._critical_flash_label,
                      font=f,
                      fill=(*bgr_to_rgb(label_col), int(255 * alpha)))

        pil_draw(frame, _draw)
        return frame

    def _draw_ghost_arrow(self, frame, square_mapper):
        """Draw a semi-transparent green arrow showing what the engine would have played
        instead of the last move. Only shown when the engine's preferred move differs
        from what was actually played, AND when YOLO confirms a piece exists on the
        source square (prevents phantom arrows from stale Stockfish positions)."""
        if self.bsm.last_move is None or square_mapper.pt.board_size is None:
            return frame
        if self._engine_preferred_move is None:
            return frame
        # Skip if the played move already matched the engine's top choice
        if self._engine_preferred_move == self.bsm.last_move:
            return frame

        # Source-square validation: suppress the arrow when YOLO sees no piece
        # on the square Stockfish wants to move from — this eliminates arrows
        # that originate from empty squares due to stale analysis positions.
        if self._last_detected_state:
            from_sq_name = chess.square_name(self._engine_preferred_move.from_square)
            if from_sq_name not in self._last_detected_state:
                return frame

        p1 = tuple(int(v) for v in self._square_center(self._engine_preferred_move.from_square, square_mapper))
        p2 = tuple(int(v) for v in self._square_center(self._engine_preferred_move.to_square, square_mapper))

        overlay = frame.copy()
        cv2.arrowedLine(overlay, p1, p2, C_PANEL_BG,  7, cv2.LINE_AA, tipLength=0.22)  # shadow
        cv2.arrowedLine(overlay, p1, p2, C_GREEN,      3, cv2.LINE_AA, tipLength=0.22)  # green
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
        return frame

    def _draw_move_arrow(self, frame, square_mapper):
        """Draw a gold arrow from the last move's from-square to to-square."""
        if self.bsm.last_move is None or square_mapper.pt.board_size is None:
            return frame
        p1 = tuple(int(v) for v in self._square_center(self.bsm.last_move.from_square, square_mapper))
        p2 = tuple(int(v) for v in self._square_center(self.bsm.last_move.to_square, square_mapper))
        # Shadow pass (dark, thick)
        cv2.arrowedLine(frame, p1, p2, C_PANEL_BG, 8, cv2.LINE_AA, tipLength=0.25)
        # Main arrow in gold
        cv2.arrowedLine(frame, p1, p2, C_GOLD, 4, cv2.LINE_AA, tipLength=0.25)
        return frame

    def _draw_piece_distribution(self, frame, detected_board_state=None):
        """
        Bottom-centre mini-board heatmap showing the current spatial distribution
        of all pieces.

        Data source priority:
          1. detected_board_state  — live YOLO detections for the current frame
             (a dict {square_name: piece_type_str}, e.g. {'e4': 'white-pawn'})
          2. self.bsm.chess_board  — confirmed engine state (fallback when no
             detection data is available, e.g. first frame or during occlusion)

        Using the detected state means the panel tracks what the camera actually
        sees — including pawns that have moved before the BSM commits a move, and
        piece counts that reflect live captures rather than the move log.

        Layout (top → bottom inside the panel):
          • "Distribution" header  +  small "live" / "engine" source badge
          • 8×8 mini board — light/dark squares, chess.com palette
          • Occupied squares filled: warm amber (white) or cool teal (black)
          • Gaussian glow halo per piece — soft influence-zone effect
          • Unicode piece symbols centred in each occupied square
          • File (a–h) and rank (8–1) axis labels
        """
        fh, fw = frame.shape[:2]

        sq    = 18          # pixels per mini square
        brd   = sq * 8      # 144 px board
        lpad  = 13          # rank-label column width
        bpad  = 8           # inner padding
        hdr_h = 22          # header height

        pw = lpad + brd + bpad * 2
        ph = hdr_h + bpad + brd + lpad + bpad

        px = fw // 2 - pw // 2
        py = fh - ph - 8

        # ── Build unified pieces list: (col_i, row_i, is_white, sym) ─────────
        # col_i  : 0=a … 7=h
        # row_i  : 0=rank-8 … 7=rank-1  (top→bottom in standard view)
        pieces = []   # (col_i, row_i, is_white, sym)

        if detected_board_state:
            # Live YOLO detections — reflects the actual physical board state
            for sq_name, pt in detected_board_state.items():
                try:
                    col_i = ord(sq_name[0]) - ord('a')   # 0-7
                    rank  = int(sq_name[1]) - 1           # convert to 0-based (0-7)
                    row_i = 7 - rank                      # 0-7 (rank8→row0)
                    if not (0 <= col_i <= 7 and 0 <= row_i <= 7):
                        continue
                    is_white = pt.startswith('white')
                    sym      = PIECE_SYM.get(pt, '?')
                    pieces.append((col_i, row_i, is_white, sym))
                except (IndexError, ValueError):
                    continue
            src_label = 'live'
        else:
            # Fallback: confirmed engine state (starting position if no moves yet)
            for chess_sq in chess.SQUARES:
                piece = self.bsm.chess_board.piece_at(chess_sq)
                if piece is None:
                    continue
                col_i    = chess.square_file(chess_sq)
                row_i    = 7 - chess.square_rank(chess_sq)
                is_white = piece.color == chess.WHITE
                sym      = piece.unicode_symbol()
                pieces.append((col_i, row_i, is_white, sym))
            src_label = 'engine'

        # ── Panel background ──────────────────────────────────────────────────
        ov = frame.copy()
        draw_rounded_rect(ov, (px, py), (px + pw, py + ph), C_PANEL_BG, radius=10)
        cv2.addWeighted(ov, 0.88, frame, 0.12, 0, frame)
        cv2.rectangle(frame, (px + 10, py), (px + pw - 10, py + 3), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph),
                          C_PANEL_BORDER, radius=10, thickness=1)

        # Board top-left pixel (origin = a8 square)
        bx = px + bpad + lpad
        by = py + hdr_h + bpad

        # ── Board base squares ────────────────────────────────────────────────
        _LIGHT = (200, 178, 140)
        _DARK  = ( 78,  55,  34)
        _W_COL = (100, 170, 245)   # warm amber — white pieces
        _B_COL = (165,  95,  42)   # cool teal  — black pieces

        for row in range(8):
            for col in range(8):
                sx, sy = bx + col * sq, by + row * sq
                cv2.rectangle(frame,
                              (sx, sy), (sx + sq - 1, sy + sq - 1),
                              _LIGHT if (row + col) % 2 == 0 else _DARK, -1)

        # ── Gaussian glow pass (influence zones) ─────────────────────────────
        glow = np.zeros((fh, fw, 3), dtype=np.float32)
        for col_i, row_i, is_white, _ in pieces:
            cx = bx + col_i * sq + sq // 2
            cy = by + row_i * sq + sq // 2
            cv2.circle(glow, (cx, cy), sq + 5,
                       _W_COL if is_white else _B_COL, -1, cv2.LINE_AA)

        glow_b = cv2.GaussianBlur(glow, (0, 0), sq * 0.55)
        roi    = np.s_[by:by + brd, bx:bx + brd]
        frame[roi] = np.clip(
            frame[roi].astype(np.float32) + glow_b[roi] * 0.30, 0, 255
        ).astype(np.uint8)

        # ── Piece fills (on top of glow) ──────────────────────────────────────
        for col_i, row_i, is_white, _ in pieces:
            sx, sy = bx + col_i * sq, by + row_i * sq
            cv2.rectangle(frame, (sx, sy), (sx + sq - 1, sy + sq - 1),
                          _W_COL if is_white else _B_COL, -1)

        # ── Labels and piece symbols via Pillow ───────────────────────────────
        def _draw(draw, _):
            draw.text((px + bpad + 2, py + 5), "Distribution",
                      font=_font(11, bold=True), fill=bgr_to_rgb(C_MUTED))
            # Tiny source badge (right-aligned in header)
            badge_col = bgr_to_rgb(C_GREEN) if src_label == 'live' else bgr_to_rgb(C_MUTED)
            draw.text((px + pw - 38, py + 6), src_label,
                      font=_font(9, bold=True), fill=badge_col)

            f_lbl   = _font(9)
            f_piece = _font(max(9, sq - 5))

            # Rank labels (8 … 1, top to bottom)
            for r in range(8):
                draw.text((px + bpad + 1, by + r * sq + sq // 2 - 5),
                          str(8 - r), font=f_lbl, fill=bgr_to_rgb(C_MUTED))

            # File labels (a … h, left to right)
            for c in range(8):
                draw.text((bx + c * sq + sq // 2 - 3, by + brd + 2),
                          chr(ord('a') + c), font=f_lbl, fill=bgr_to_rgb(C_MUTED))

            # Piece Unicode symbols centred in each occupied square
            for col_i, row_i, is_white, sym in pieces:
                sx   = bx + col_i * sq
                sy   = by + row_i * sq
                tw   = f_piece.getlength(sym)
                tcol = (240, 228, 208) if is_white else (22, 18, 14)
                draw.text((sx + (sq - tw) / 2, sy + 1), sym,
                          font=f_piece, fill=tcol)

        pil_draw(frame, _draw)
        return frame

    def _draw_board_coordinates(self, frame, square_mapper):
        """Draw file (a–h) and rank (1–8) labels at the board edges, chess.com style.
        File letters sit in the bottom-right corner of each bottom-row square;
        rank numbers sit in the top-left corner of each left-column square."""
        if square_mapper.pt.board_size is None:
            return frame

        sq_px = square_mapper.pt.board_size / 8
        labels = []  # list of (fx, fy, text)

        for col in range(8):
            # Bottom-right of each square in the bottom row (row 7)
            bx = (col + 0.88) * sq_px
            by = 7.82 * sq_px
            fx, fy = square_mapper.pt.inverse_transform_point((bx, by))
            letter = chr(ord('a') + col) if square_mapper.white_at_bottom else chr(ord('h') - col)
            labels.append((fx, fy, letter))

        for row in range(8):
            # Top-left of each square in the left column (col 0)
            bx = 0.05 * sq_px
            by = (row + 0.08) * sq_px
            fx, fy = square_mapper.pt.inverse_transform_point((bx, by))
            number = str(8 - row) if square_mapper.white_at_bottom else str(row + 1)
            labels.append((fx, fy, number))

        def _draw(draw, _):
            f = _font(10, bold=True)
            for fx, fy, text in labels:
                # Subtle dark shadow for readability on any square color
                draw.text((fx + 1, fy + 1), text, font=f, fill=(10, 10, 10, 200))
                draw.text((fx,     fy    ), text, font=f, fill=(*bgr_to_rgb(C_CREAM), 210))

        pil_draw(frame, _draw)
        return frame

    def draw_captured_pieces(self, frame):
        """Render a captured-pieces panel below the stats overlay (~y=110)."""
        w_score, b_score = self._material_advantage()
        adv = w_score - b_score

        # Layout
        px, py = 10, 110
        pw, ph = 230, 84
        radius  = 10

        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, radius)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, radius, thickness=1)

        # Sort captured pieces by value (highest first) for compact display
        def _row_sym(cap_list, color):
            by_val = sorted(cap_list,
                            key=lambda pt: -PIECE_VALUES.get(pt, 0))
            return ''.join(_CAP_SYM[color].get(pt, '?') for pt in by_val)

        black_row = _row_sym(self._eff_cap_white, chess.BLACK)  # white captured black pieces
        white_row = _row_sym(self._eff_cap_black, chess.WHITE)  # black captured white pieces

        # Material advantage badge text
        if adv > 0:
            badge = f'+{adv}'
            badge_col = C_EVAL_WHITE
        elif adv < 0:
            badge = f'{adv}'
            badge_col = C_EVAL_BLACK
        else:
            badge = '='
            badge_col = C_MUTED

        def _draw(draw, _):
            draw.text((px + 10, py + 6),  "Captured Pieces", font=_font(11, bold=True),
                      fill=bgr_to_rgb(C_MUTED))
            # Black pieces row (captured by white) — shown in white-indicator row
            draw.text((px + 10, py + 26), '♟' if not black_row else black_row,
                      font=_font(15), fill=bgr_to_rgb(C_WHITE_IND))
            # White pieces row (captured by black) — shown in black-indicator row
            draw.text((px + 10, py + 50), '♙' if not white_row else white_row,
                      font=_font(15), fill=bgr_to_rgb(C_BLACK_IND))
            # Material badge
            bw = 26
            bx2 = px + pw - 12
            draw.text((bx2 - bw // 2, py + 34), badge, font=_font(14, bold=True),
                      fill=bgr_to_rgb(badge_col))

        pil_draw(frame, _draw)
        return frame

    def _draw_last_move_highlight(self, frame, square_mapper):
        if self.bsm.last_move is None or square_mapper.pt.board_size is None:
            return frame
        overlay = frame.copy()
        # From-square: amber tint; to-square: blue tint
        cv2.fillPoly(overlay,
                     [self._square_corners(self.bsm.last_move.from_square, square_mapper)],
                     (0, 140, 255))
        cv2.fillPoly(overlay,
                     [self._square_corners(self.bsm.last_move.to_square, square_mapper)],
                     (255, 140, 0))
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        return frame

    def _draw_eval_bar(self, frame, eval_score):
        if eval_score is None:
            return frame

        fh = frame.shape[0]
        bx, by = frame.shape[1] - 30, 10
        bw, bh = 20, fh - 20
        radius = 6

        # Background track
        draw_rounded_rect(frame, (bx, by), (bx + bw, by + bh), C_PANEL_BG, radius)

        clamped    = max(-500, min(500, eval_score))
        white_frac = (clamped + 500) / 1000.0
        white_h    = int(bh * white_frac)
        black_h    = bh - white_h
        split_y    = by + black_h

        # Black fill (top)
        if black_h > radius:
            draw_rounded_rect(frame, (bx, by), (bx + bw, split_y), C_EVAL_BLACK, radius)
        # White fill (bottom)
        if white_h > radius:
            draw_rounded_rect(frame, (bx, split_y), (bx + bw, by + bh), C_EVAL_WHITE, radius)

        # Thin green divider line at the split point
        cv2.line(frame, (bx, split_y), (bx + bw, split_y), C_GREEN, 2)

        # Border
        draw_rounded_rect(frame, (bx, by), (bx + bw, by + bh), C_PANEL_BORDER, radius, thickness=1)

        label = (f"{eval_score / 100:+.1f}"
                 if abs(eval_score) < 10000
                 else ("M+" if eval_score > 0 else "M-"))

        # Pill background above the bar for readability
        pill_w, pill_h = 36, 18
        pill_x = bx + bw // 2 - pill_w // 2
        pill_y = by - pill_h - 4
        draw_rounded_rect(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
                          C_PANEL_BG, radius=4)
        draw_rounded_rect(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h),
                          C_PANEL_BORDER, radius=4, thickness=1)

        def _draw(draw, _):
            f = _font(12, bold=True)
            tw = f.getlength(label)
            draw.text((pill_x + (pill_w - tw) / 2, pill_y + 2),
                      label, font=f, fill=bgr_to_rgb(C_TEXT))

        pil_draw(frame, _draw)
        return frame

    def _draw_move_history(self, frame):
        self._rebuild_san_cache()
        if not self._san_cache:
            return frame

        # Build (move_num, white_san, white_annot, black_san, black_annot) tuples
        annots = self._move_annotations  # may be shorter than _san_cache if Stockfish is slow
        pairs = []
        for i in range(0, len(self._san_cache), 2):
            w_san  = self._san_cache[i]
            b_san  = self._san_cache[i + 1] if i + 1 < len(self._san_cache) else ""
            w_sym, w_col = annots[i]     if i     < len(annots) else ('', C_TEXT)
            b_sym, b_col = annots[i + 1] if i + 1 < len(annots) else ('', C_TEXT)
            pairs.append((i // 2 + 1, w_san, w_sym, w_col, b_san, b_sym, b_col))

        visible  = pairs[-8:]
        row_h    = 20
        header_h = 28
        px       = frame.shape[1] - 210  # slightly wider to fit annotation symbols
        py       = 10
        pw       = 178
        ph       = header_h + len(visible) * row_h + 8
        radius   = 10

        overlay = frame.copy()
        draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, radius)
        cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

        cv2.rectangle(frame, (px + radius, py), (px + pw - radius, py + 3), C_GREEN, -1)
        draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, radius, thickness=1)

        last_row_y = py + header_h + (len(visible) - 1) * row_h
        draw_rounded_rect(frame,
                          (px + 4, last_row_y - 2),
                          (px + pw - 4, last_row_y + row_h - 2),
                          (45, 42, 36), radius=4)

        # Map visible pair index → absolute move indices (for move-time lookup)
        pair_start = max(0, len(pairs) - 8)

        def _draw(draw, _):
            draw.text((px + 10, py + 6), "Move History", font=_font(13, bold=True),
                      fill=bgr_to_rgb(C_MUTED))
            for i, (num, w_san, w_sym, w_col, b_san, b_sym, b_col) in enumerate(visible):
                ry = py + header_h + i * row_h
                is_last = (i == len(visible) - 1)
                num_col    = bgr_to_rgb(C_MUTED)
                w_move_col = bgr_to_rgb(C_GOLD if is_last else C_TEXT)
                b_move_col = bgr_to_rgb(C_GOLD if is_last else C_TEXT)
                draw.text((px + 8,  ry), f"{num:>2}.", font=_font(12), fill=num_col)
                draw.text((px + 36, ry), w_san, font=_font(12, bold=is_last), fill=w_move_col)
                if w_sym:
                    draw.text((px + 36 + _font(12).getlength(w_san) + 2, ry),
                              w_sym, font=_font(11, bold=True),
                              fill=(*bgr_to_rgb(w_col), 255))
                draw.text((px + 108, ry), b_san, font=_font(12, bold=is_last), fill=b_move_col)
                if b_sym:
                    draw.text((px + 108 + _font(12).getlength(b_san) + 2, ry),
                              b_sym, font=_font(11, bold=True),
                              fill=(*bgr_to_rgb(b_col), 255))
                # Move-time badge on the current (highlighted) row
                if is_last:
                    pair_idx = pair_start + i
                    w_move_idx = pair_idx * 2
                    b_move_idx = pair_idx * 2 + 1
                    t_font = _font(9)
                    t_col  = (*bgr_to_rgb(C_MUTED), 180)
                    if w_move_idx < len(self._move_times):
                        t = self._move_times[w_move_idx]
                        draw.text((px + 36, ry + 9), f"{t:.0f}s",
                                  font=t_font, fill=t_col)
                    if b_move_idx < len(self._move_times):
                        t = self._move_times[b_move_idx]
                        draw.text((px + 108, ry + 9), f"{t:.0f}s",
                                  font=t_font, fill=t_col)

        pil_draw(frame, _draw)
        return frame


class ChessTracker:
    """
    ChessTracker that uses YOLO for initial board detection + OpenCV pattern detector for precise corners
    """
    def __init__(self, model_path, chessboard_model_path, seg_model_path=None):
        """
        Initialize ChessTracker
        Args:
            model_path (str): Path to YOLO model for piece detection
            chessboard_model_path (str): Path to YOLO model for chessboard detection
            seg_model_path (str, optional): Path to a YOLO-seg model that detects
                AND segments pieces in one pass. When given, replaces the
                detect (model_path) + SAM2-refine pipeline in main()'s per-frame
                loop with a single inference call.
        """
        self.model = YOLO(model_path)
        self.chessboard_model = YOLO(chessboard_model_path)

        if seg_model_path and os.path.exists(seg_model_path):
            self.seg_model = YOLO(seg_model_path)
            print(f"ChessTracker: seg model loaded ({seg_model_path})")
        else:
            self.seg_model = None

        self.corner_detector     = LSDChessboardCornerDetector()
        self.pattern_detector    = PatternChessboardCornerDetector()
        self._last_good_corners  = None   # cache for frames where pattern fails

        # Initialize tracker and perspective transformer
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=90,           # 3 s at 30 fps — keep tracks alive through motor movement
            minimum_matching_threshold=0.3  # low IoU ok: pieces physically move to new squares
        )
        self.perspective_transformer = PerspectiveTransformer()
        self.square_mapper = SquareMapper(self.perspective_transformer)
        self.board_state_manager = BoardStateManager(stability_threshold=4, debug=False)
        stockfish_path = '/usr/games/stockfish'
        self.game_engine = GameStateEngine(
            self.board_state_manager,
            stockfish_path=stockfish_path if os.path.exists(stockfish_path) else None,
            fps=30.0   # updated via tracker.game_engine._fps = fps after video load
        )
        self.speed_distance_estimator = SpeedAndDistanceEstimator()

        # SAM2 base-point refiner (optional — fails gracefully if unavailable)
        try:
            self.sam_refiner = SAM3Refiner()
        except Exception as e:
            print(f"ChessTracker: SAM3Refiner init failed ({e}); bbox fallback active")
            self.sam_refiner = None

        # Cache: track_id → decided piece_type for king/queen disambiguation.
        # Once SAM2 (or the heuristic) has resolved a track, we reuse that
        # decision instead of re-running SAM2 on every frame.
        self._royalty_track_cache: dict = {}   # {track_id: piece_type_str}

    def detect_chessboard_bbox(self, frame):
        """
        Detect chessboard bounding box using YOLO.
        Guards against false positives from digital screens / tablets by
        requiring a minimum area and a roughly-square aspect ratio.
        Args:
            frame (numpy.ndarray): Video frame
        Returns:
            list or None: Bounding box [x1, y1, x2, y2] or None if detection fails
        """
        fh, fw = frame.shape[:2]
        # Physical board must be at least 20 % of frame area and roughly square
        MIN_AREA  = fw * fh * 0.20   # ≥ 20 % of frame
        MIN_SIDE  = min(fw, fh) * 0.25
        MAX_RATIO = 1.6              # width/height must stay below this

        results = self.chessboard_model.predict(frame, conf=0.5, verbose=False)

        best = None
        best_area = 0
        if results and results[0].boxes:
            for box in results[0].boxes.xyxy:
                x1, y1, x2, y2 = box.int().tolist()
                w, h = x2 - x1, y2 - y1
                area = w * h
                ratio = max(w, h) / max(min(w, h), 1)
                # Reject tiny or very non-square candidates (tablet digital board)
                if w < MIN_SIDE or h < MIN_SIDE:
                    print(f"  Board candidate rejected (too small): [{x1},{y1},{x2},{y2}]")
                    continue
                if area < MIN_AREA:
                    print(f"  Board candidate rejected (area {area:.0f} < {MIN_AREA:.0f}): [{x1},{y1},{x2},{y2}]")
                    continue
                if ratio > MAX_RATIO:
                    print(f"  Board candidate rejected (aspect {ratio:.2f}): [{x1},{y1},{x2},{y2}]")
                    continue
                if area > best_area:
                    best_area = area
                    best = [x1, y1, x2, y2]

        if best:
            print(f"YOLO detected chessboard bbox: {best}")
            return best

        print("Warning: YOLO chessboard detection failed")
        return None

    def prepare_tracking(self, first_frame, debug=False, manual_bbox=None, headless=False):
        """
        Prepare tracking by detecting the chessboard using YOLO + OpenCV pattern detector.
        Args:
            first_frame (numpy.ndarray): First frame of the video
            debug (bool): Enable debug output and visualization
            manual_bbox (list, optional): Manually provided [x1, y1, x2, y2] if auto-detection fails
            headless (bool): Set True when running without display (cloud/server environment)
        """
        print("Starting board detection with YOLO + OpenCV chessboard pattern detector...")

        # Step 1: Get a rough bounding box from YOLO to constrain the search region
        if manual_bbox is None:
            yolo_bbox = self.detect_chessboard_bbox(first_frame)
            if yolo_bbox is None:
                print("YOLO detection failed, running pattern detector on full frame...")
        else:
            yolo_bbox = manual_bbox
            print(f"Using manual bbox: {yolo_bbox}")

        # Step 2: Detect corners — scan up to 300 frames for a full n=7 pattern
        # (early frames may have a hand blocking inner corners)
        print("Trying inner-corner pattern detector (findChessboardCornersSB)...")
        corners = None
        _all_frames = [first_frame]
        if hasattr(self, '_init_frames') and self._init_frames:
            _all_frames = self._init_frames
        for _probe in _all_frames[:300]:
            # Require full 7×7 grid for initial calibration — ensures corners
            # cover the whole board, not just a partial visible region
            _c = self.pattern_detector.detect_board_corners(_probe, min_n=7)
            if _c is not None:
                corners = _c
                break
        if corners is None:
            print("Pattern detector failed on all probed frames — trying LSD...")
            corners = self.corner_detector.detect_board_corners(first_frame, yolo_bbox)

        if corners is None:
            print("LSD failed — falling back to YOLO bbox rectangle...")
            if yolo_bbox is not None:
                x1, y1, x2, y2 = yolo_bbox
                corners = np.array([
                    [x1, y1], [x2, y1], [x1, y2], [x2, y2]
                ], dtype=np.float32)
            else:
                raise ValueError("Could not detect chessboard corners!")

        self._last_good_corners = corners.copy()

        self.perspective_transformer.set_board_corners(corners)
        self.perspective_transformer.estimate_board_perspective()

        print(f"\n✓ Board detected successfully!")
        print(f"Corners: {corners}")

        vis_frame = first_frame.copy()
        corner_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        corner_labels = ["TL", "TR", "BL", "BR"]

        if yolo_bbox:
            bx1, by1, bx2, by2 = [int(v) for v in yolo_bbox]
            cv2.rectangle(vis_frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
            cv2.putText(vis_frame, "YOLO BBox", (bx1, by1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for i, corner in enumerate(corners):
            cv2.circle(vis_frame, tuple(corner.astype(int)), 10, corner_colors[i], -1)
            cv2.putText(vis_frame, corner_labels[i],
                        tuple((corner + 15).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, corner_colors[i], 2)

        cv2.polylines(vis_frame, [corners[[0, 1, 3, 2]].astype(np.int32)], True, (0, 255, 0), 3)

        if not headless:
            cv2.imshow("Board Detection", vis_frame)
            print("\nPress any key to continue...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            output_path = '/teamspace/studios/this_studio/board_detection.jpg'
            cv2.imwrite(output_path, vis_frame)
            print(f"Board detection visualization saved to: {output_path}")

    def update_board_corners(self, frame):
        """
        Update the chessboard corners for the current frame using YOLO + pattern detector.
        Args:
            frame (numpy.ndarray): Current video frame
        """
        yolo_bbox = self.detect_chessboard_bbox(frame)

        if yolo_bbox is None:
            if self.perspective_transformer.board_corners is not None:
                print("Warning: YOLO detection failed, using previous corners")
            else:
                print("Warning: No board detection available")
            return

        # Try pattern detector first (works mid-game), then LSD fallback
        corners = self.pattern_detector.detect_board_corners(frame)
        if corners is None:
            corners = self.corner_detector.detect_board_corners(frame, yolo_bbox)

        if corners is not None:
            self._last_good_corners = corners.copy()
            self.perspective_transformer.set_board_corners(corners)
            self.perspective_transformer.estimate_board_perspective()
        else:
            # Both detectors failed — keep the last good corners (cache)
            print("Warning: corner detection failed, using cached corners")

    def highlight_possible_moves(self, frame, piece_position, board_state):
        """
        Highlight possible moves for a piece on the frame
        """
        rank, file = self.convert_to_chess_notation(piece_position)
        if rank is None or file is None:
            print(f"Skipping invalid position: {piece_position}")
            return frame

        try:
            square = chess.parse_square(f"{file}{rank}")
        except ValueError:
            print(f"Invalid square: {file}{rank}")
            return frame

        # Generate possible moves
        possible_moves = [move for move in board_state.legal_moves if move.from_square == square]

        for move in possible_moves:
            to_rank, to_file = chess.square_rank(move.to_square), chess.square_file(move.to_square)
            to_x, to_y = self.convert_to_frame_coordinates(to_rank, chr(ord('a') + to_file))
            if to_x is not None and to_y is not None:
                cv2.circle(frame, (to_x, to_y), 10, (0, 255, 0), -1)

        return frame

    def convert_to_chess_notation(self, piece_position):
        """
        Convert frame coordinates to chess notation
        """
        x, y = piece_position

        if self.perspective_transformer.board_corners is None:
            return None, None

        # Transform the position to the perspective-aligned board space
        transformed_point = self.perspective_transformer.transform_point((x, y))

        # Get board size from transformation
        corners = self.perspective_transformer.board_corners
        width = max(
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[2] - corners[3])
        )
        
        board_size = 8
        square_size = width / board_size
        
        col = int(transformed_point[0] // square_size)
        row = int(transformed_point[1] // square_size)

        # Validate the column and row
        if col < 0 or col >= board_size or row < 0 or row >= board_size:
            print(f"Invalid board coordinates: col={col}, row={row}")
            return None, None

        # Convert to chess notation
        file = chr(ord('a') + col)
        rank = 8 - row
        return rank, file

    def convert_to_frame_coordinates(self, rank, file):
        """
        Convert chess rank and file to frame coordinates
        """
        if self.perspective_transformer.board_corners is None:
            return None, None

        # Validate file type
        if not isinstance(file, str):
            print(f"Invalid file type: {file}")
            return None, None

        # Map rank and file to board space
        col = ord(file) - ord('a')
        row = 8 - rank

        # Validate the column and row
        if col < 0 or col >= 8 or row < 0 or row >= 8:
            print(f"Invalid rank or file: rank={rank}, file={file}")
            return None, None

        # Get board size
        corners = self.perspective_transformer.board_corners
        width = max(
            np.linalg.norm(corners[0] - corners[1]),
            np.linalg.norm(corners[2] - corners[3])
        )
        
        board_size = 8
        square_size = width / board_size

        # Calculate the center of the square
        x_center = (col + 0.5) * square_size
        y_center = (row + 0.5) * square_size

        # Transform the point back to frame coordinates
        frame_coordinates = self.perspective_transformer.inverse_transform_point((x_center, y_center))

        return frame_coordinates

    def smooth_tracks(self, tracks, smoothing_factor=0.7):
        """
        Smooth track positions across frames to reduce jittering
        """
        smoothed_tracks = {
            'white_pieces': [],
            'black_pieces': [],
            'board': []
        }
        
        prev_white_positions = {}
        prev_black_positions = {}
        
        for frame_num in range(len(tracks['white_pieces'])):
            smoothed_tracks['white_pieces'].append({})
            smoothed_tracks['black_pieces'].append({})
            smoothed_tracks['board'].append({})
            
            # Smooth white pieces
            for track_id, piece in tracks['white_pieces'][frame_num].items():
                current_pos = piece['position']
                
                if track_id in prev_white_positions:
                    smooth_pos = [
                        smoothing_factor * prev_white_positions[track_id][i] +
                        (1 - smoothing_factor) * current_pos[i]
                        for i in range(len(current_pos))
                    ]
                else:
                    smooth_pos = current_pos
                
                smoothed_piece = piece.copy()
                smoothed_piece['position'] = smooth_pos
                smoothed_tracks['white_pieces'][frame_num][track_id] = smoothed_piece
                
                prev_white_positions[track_id] = smooth_pos
            
            # Smooth black pieces
            for track_id, piece in tracks['black_pieces'][frame_num].items():
                current_pos = piece['position']
                
                if track_id in prev_black_positions:
                    smooth_pos = [
                        smoothing_factor * prev_black_positions[track_id][i] +
                        (1 - smoothing_factor) * current_pos[i]
                        for i in range(len(current_pos))
                    ]
                else:
                    smooth_pos = current_pos
                
                smoothed_piece = piece.copy()
                smoothed_piece['position'] = smooth_pos
                smoothed_tracks['black_pieces'][frame_num][track_id] = smoothed_piece
                
                prev_black_positions[track_id] = smooth_pos
        
        return smoothed_tracks

    def _enforce_royalty_constraints(self, piece_dict: dict, side: str,
                                      frame=None) -> dict:
        """
        Enforce chess rules: at most 1 king and 1 queen per side per frame.

        Uses a track-ID cache (_royalty_track_cache) so SAM3 is called only
        once per ambiguous track pair — all subsequent frames re-use the cached
        decision without any model inference.

        Strategy (in order of priority):
          1. Apply any already-cached king/queen decisions first.
          2. Too many kings → keep the highest-confidence king; re-label rest.
          3. Multiple queens, no king detected:
             a. Check if any of the queen track-IDs is cached as king.
             b. Otherwise run SAM3 mask-area once, cache result, pick smallest
                area as king (shorter piece).
             c. Fallback: shortest bbox height → king.
          4. Multiple queens, king present → keep highest-confidence queen,
             re-label extras as bishop.

        Args:
            piece_dict : {track_id: {'bbox': …, 'piece_type': …, 'conf': …}}
            side       : 'white' | 'black'
            frame      : BGR frame for SAM3 (only used on first encounter)
        Returns modified copy of piece_dict.
        """
        king_t   = f'{side}-king'
        queen_t  = f'{side}-queen'
        bishop_t = f'{side}-bishop'

        result = {tid: info.copy() for tid, info in piece_dict.items()}

        # ── Step 0: apply cached decisions ──────────────────────────────────
        for tid, info in result.items():
            if tid in self._royalty_track_cache:
                cached = self._royalty_track_cache[tid]
                if info['piece_type'] != cached:
                    result[tid]['piece_type'] = cached

        def _refresh():
            kings  = [(tid, info) for tid, info in result.items()
                      if info['piece_type'] == king_t]
            queens = [(tid, info) for tid, info in result.items()
                      if info['piece_type'] == queen_t]
            return kings, queens

        def _conf(info):
            return info.get('conf', 0.5)

        def _bbox_h(info):
            b = info['bbox']
            return b[3] - b[1]

        kings, queens = _refresh()

        # ── Step 1: Too many kings ───────────────────────────────────────────
        if len(kings) > 1:
            kings_sorted = sorted(kings, key=lambda x: _conf(x[1]), reverse=True)
            keep_tid = kings_sorted[0][0]
            self._royalty_track_cache[keep_tid] = king_t
            for tid, info in kings_sorted[1:]:
                result[tid]['piece_type'] = queen_t
                self._royalty_track_cache[tid] = queen_t
                print(f"  [royalty] {side} extra king t{tid} conf={_conf(info):.2f}"
                      f" → queen  (cached)")
            kings, queens = _refresh()

        # ── Step 2: Multiple queens, no king ────────────────────────────────
        if len(queens) > 1 and len(kings) == 0:
            # 2a: check if any queen track is already cached as king
            king_tid = next(
                (tid for tid, _ in queens
                 if self._royalty_track_cache.get(tid) == king_t),
                None
            )

            # 2b: SAM3 morphology (only if not cached yet)
            if king_tid is None:
                uncached = [(tid, info) for tid, info in queens
                            if tid not in self._royalty_track_cache]
                if (uncached and frame is not None
                        and self.sam_refiner is not None
                        and self.sam_refiner.available):
                    bboxes    = [info['bbox'] for _, info in uncached]
                    ptypes    = [info['piece_type'] for _, info in uncached]
                    features  = self.sam_refiner.get_mask_features(
                        frame, bboxes, piece_types=ptypes)
                    areas    = [f['area'] for f in features]
                    if any(a > 0 for a in areas):
                        min_idx  = int(np.argmin(areas))
                        king_tid = uncached[min_idx][0]
                        # Cache all decisions in this ambiguous group
                        for i, (tid, _) in enumerate(uncached):
                            decision = king_t if i == min_idx else queen_t
                            self._royalty_track_cache[tid] = decision
                        print(f"  [royalty] SAM3 {side} king → t{king_tid} "
                              f"area={areas[min_idx]}  (cached for all "
                              f"{len(uncached)} tracks)")

            # 2c: bbox-height fallback
            if king_tid is None:
                queens_by_h = sorted(queens, key=lambda x: _bbox_h(x[1]))
                king_tid = queens_by_h[0][0]
                for i, (tid, _) in enumerate(queens_by_h):
                    decision = king_t if i == 0 else queen_t
                    self._royalty_track_cache[tid] = decision
                print(f"  [royalty] bbox-h {side} king → t{king_tid} "
                      f"h={_bbox_h(dict(queens)[king_tid]):.0f}px  (cached)")

            result[king_tid]['piece_type'] = king_t
            kings, queens = _refresh()

        # ── Step 3: Multiple queens with a king ─────────────────────────────
        if len(queens) > 1 and len(kings) >= 1:
            queens_sorted = sorted(queens, key=lambda x: _conf(x[1]), reverse=True)
            keep_tid = queens_sorted[0][0]
            self._royalty_track_cache[keep_tid] = queen_t
            for tid, info in queens_sorted[1:]:
                result[tid]['piece_type'] = bishop_t
                self._royalty_track_cache[tid] = bishop_t
                print(f"  [royalty] {side} extra queen t{tid} conf={_conf(info):.2f}"
                      f" → bishop  (cached)")

        return result

    def detect_frames(self, frames):
        """Batch detection of frames.
        Processes one frame at a time and moves mask tensors to CPU immediately
        so GPU memory stays bounded regardless of video length."""
        import torch
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size], conf=0.15)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """Track chess pieces across frames.

        Piece detections whose centre falls outside the locked-in board bbox
        are discarded — this removes false positives from the tablet/digital
        screen visible on the right side of the frame.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        # Board region from the perspective transformer (set during prepare_tracking).
        # Build an axis-aligned guard box from the four corners.
        board_guard = None
        if self.perspective_transformer.board_corners is not None:
            corners = self.perspective_transformer.board_corners
            gx1 = int(corners[:, 0].min()) - 10
            gy1 = int(corners[:, 1].min()) - 10
            gx2 = int(corners[:, 0].max()) + 10
            gy2 = int(corners[:, 1].max()) + 10
            board_guard = (gx1, gy1, gx2, gy2)
            print(f"Board guard bbox for piece clipping: {board_guard}")

        tracks = {
            "white_pieces": [],
            "black_pieces": [],
            "board": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names

            detection_supervision = sv.Detections.from_ultralytics(detection)
            # Cross-class NMS: remove duplicate boxes where YOLO fires two
            # different class labels on the same physical piece (e.g., a bishop
            # also detected as a knight on an identical bounding box).
            # YOLO's built-in NMS only handles same-class overlaps; this pass
            # handles cross-class overlaps before they reach ByteTrack.
            detection_supervision = _cross_class_nms(detection_supervision, iou_threshold=0.5)
            detections_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["white_pieces"].append({})
            tracks["black_pieces"].append({})
            tracks["board"].append({})

            for frame_detection in detections_with_tracks:
                bbox     = frame_detection[0].tolist()
                conf     = float(frame_detection[2]) if frame_detection[2] is not None else 0.5
                cls_id   = frame_detection[3]
                track_id = frame_detection[4]

                # ── Board-region guard: drop pieces outside the physical board ──
                if board_guard is not None:
                    cx = (bbox[0] + bbox[2]) / 2
                    cy = (bbox[1] + bbox[3]) / 2
                    gx1, gy1, gx2, gy2 = board_guard
                    if not (gx1 <= cx <= gx2 and gy1 <= cy <= gy2):
                        continue   # outside board region → discard

                piece_name = cls_names[cls_id]
                if piece_name.startswith('white'):
                    tracks['white_pieces'][frame_num][track_id] = {
                        'bbox': bbox,
                        'piece_type': piece_name,
                        'conf': conf,
                    }
                elif piece_name.startswith('black'):
                    tracks['black_pieces'][frame_num][track_id] = {
                        'bbox': bbox,
                        'piece_type': piece_name,
                        'conf': conf,
                    }
                elif piece_name == 'board':
                    tracks['board'][frame_num][track_id] = {'bbox': bbox}

            # ── Enforce 1-king / 1-queen per side using SAM2 morphology ──────
            frame_img = frames[frame_num] if frame_num < len(frames) else None
            tracks['white_pieces'][frame_num] = self._enforce_royalty_constraints(
                tracks['white_pieces'][frame_num], 'white', frame=frame_img)
            tracks['black_pieces'][frame_num] = self._enforce_royalty_constraints(
                tracks['black_pieces'][frame_num], 'black', frame=frame_img)

        if stub_path:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def add_position_to_tracks(self, tracks):
        """Add center positions to tracked objects"""
        for object_type, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    x_center = (bbox[0] + bbox[2]) / 2
                    y_center = (bbox[1] + bbox[3]) / 2
                    tracks[object_type][frame_num][track_id]['position'] = (x_center, y_center)

    def draw_piece_distance_markers(self, frame, tracks, piece_interactions):
        """
        Draw distance markers between chess pieces
        """
        CLOSE_DISTANCE = 2.0
        MEDIUM_DISTANCE = 4.0

        overlay = frame.copy()

        for interaction in piece_interactions['white_vs_black']:
            frame_num = interaction['frame']
            distance = interaction['distance']
            
            if distance <= MEDIUM_DISTANCE:
                white_pieces = tracks['white_pieces'][frame_num]
                black_pieces = tracks['black_pieces'][frame_num]
                
                if (interaction['white_id'] in white_pieces and 
                    interaction['black_id'] in black_pieces):
                    
                    white_pos = white_pieces[interaction['white_id']]['position']
                    black_pos = black_pieces[interaction['black_id']]['position']
                    
                    white_piece_type = white_pieces[interaction['white_id']].get('piece_type', 'Unknown')
                    black_piece_type = black_pieces[interaction['black_id']].get('piece_type', 'Unknown')
                    
                    white_piece_name = white_piece_type.replace('white-', 'White-').replace('-', ' ').title()
                    black_piece_name = black_piece_type.replace('black-', 'Black-').replace('-', ' ').title()
                    
                    white_piece_name_with_id = f"{white_piece_name} (ID: {interaction['white_id']})"
                    black_piece_name_with_id = f"{black_piece_name} (ID: {interaction['black_id']})"
                    
                    if distance <= CLOSE_DISTANCE:
                        line_color = (0, 0, 255)
                    else:
                        line_color = (0, 255, 255)

                    cv2.line(
                        overlay, 
                        (int(white_pos[0]), int(white_pos[1])), 
                        (int(black_pos[0]), int(black_pos[1])), 
                        line_color, 
                        1, 
                        cv2.LINE_4
                    )

                    mid_x = int((white_pos[0] + black_pos[0]) / 2)
                    mid_y = int((white_pos[1] + black_pos[1]) / 2)

                    cv2.putText(
                        overlay,
                        f"{distance:.1f} sq",
                        (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        line_color,
                        2
                    )
                    
                    cv2.putText(
                        overlay,
                        f"{white_piece_name_with_id}",
                        (int(white_pos[0]), int(white_pos[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        2
                    )
                    cv2.putText(
                        overlay,
                        f"{black_piece_name_with_id}",
                        (int(black_pos[0]), int(black_pos[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame

    def draw_statistics_overlay(self, frame, white_pieces, black_pieces):
        """Player cards showing piece counts with chess-themed styling."""
        if frame is None or frame.size == 0:
            return frame
        try:
            total = white_pieces + black_pieces
            w_pct = white_pieces / total * 100 if total else 0
            b_pct = black_pieces / total * 100 if total else 0

            px, py, pw, ph = 10, 10, 230, 96
            radius = 10

            # Semi-transparent dark panel
            overlay = frame.copy()
            draw_rounded_rect(overlay, (px, py), (px + pw, py + ph), C_PANEL_BG, radius)
            cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

            # Green left-accent bar
            cv2.rectangle(frame, (px, py + radius), (px + 4, py + ph - radius), C_GREEN, -1)
            cv2.circle(frame, (px + 2, py + radius),     2, C_GREEN, -1)
            cv2.circle(frame, (px + 2, py + ph - radius), 2, C_GREEN, -1)

            # Subtle border
            draw_rounded_rect(frame, (px, py), (px + pw, py + ph), C_PANEL_BORDER, radius, thickness=1)

            # Divider between rows
            mid_y = py + ph // 2
            cv2.line(frame, (px + 12, mid_y), (px + pw - 12, mid_y), C_PANEL_BORDER, 1)

            def draw_row(img, label, count, pct, row_y, ind_color):
                # Color indicator circle
                cv2.circle(img, (px + 18, row_y + 12), 7, ind_color, -1)
                cv2.circle(img, (px + 18, row_y + 12), 7, C_PANEL_BORDER, 1)
                # Count badge
                badge_x = px + pw - 58
                draw_rounded_rect(img, (badge_x, row_y + 2), (badge_x + 46, row_y + 22),
                                   C_GREEN, radius=4)
                return img

            frame = draw_row(frame, 'White', white_pieces, w_pct, py + 2,  C_WHITE_IND)
            frame = draw_row(frame, 'Black', black_pieces, b_pct, mid_y + 2, C_BLACK_IND)

            # Unicode text via Pillow (one round-trip for the whole panel)
            def _draw(draw, _):
                draw.text((px + 32, py + 4),    '♔ White', font=_font(15, bold=True),
                          fill=bgr_to_rgb(C_TEXT))
                draw.text((px + 32, mid_y + 4), '♚ Black', font=_font(15, bold=True),
                          fill=bgr_to_rgb(C_TEXT))
                # Clean count badges — no percentage
                f = _font(13, bold=True)
                for count, ry in [(white_pieces, py + 4), (black_pieces, mid_y + 4)]:
                    label = str(count)
                    tw = f.getlength(label)
                    draw.text((px + pw - 56 + (46 - tw) / 2, ry),
                              label, font=f, fill=bgr_to_rgb(C_PANEL_BG))

            pil_draw(frame, _draw)
        except Exception as e:
            print(f"Error in draw_statistics_overlay: {e}")
        return frame

    def draw_annotations(self, video_frames, tracks, fps):
        """Annotate video frames with piece tracking, speed, and distance"""
        output_video_frames = []

        piece_movements = self.speed_distance_estimator.estimate_piece_movement(tracks, fps)

        print("\n--- Piece Movement Statistics ---")
        for track_id, movement_data in piece_movements.items():
            print(f"Track ID: {track_id}")
            print(f"  Piece Type: {movement_data['piece_type']}")
            print(f"  Total Distance: {movement_data['total_distance']:.2f} m")
            print(f"  Max Speed: {movement_data['max_speed']:.2f} m/s\n")

        for frame_num, frame in enumerate(video_frames):
            annotated_frame = frame.copy()

            white_pieces_count = len(tracks['white_pieces'][frame_num])
            black_pieces_count = len(tracks['black_pieces'][frame_num])

            print(f"Frame {frame_num}: White Pieces: {white_pieces_count}, Black Pieces: {black_pieces_count}")

            annotated_frame = self.draw_statistics_overlay(annotated_frame, white_pieces_count, black_pieces_count)

            cv2.putText(
                annotated_frame, 
                f"FPS: {fps:.2f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )

            for track_id, piece in tracks['white_pieces'][frame_num].items():
                movement_data = piece_movements.get(track_id, {})
                self._draw_piece_annotation(
                    annotated_frame, 
                    piece['bbox'], 
                    color=(255, 255, 255),
                    track_id=track_id,
                    label=piece['piece_type'],
                    distance=movement_data.get('total_distance', 0),
                    speed=movement_data.get('max_speed', 0)
                )

            for track_id, piece in tracks['black_pieces'][frame_num].items():
                movement_data = piece_movements.get(track_id, {})
                self._draw_piece_annotation(
                    annotated_frame, 
                    piece['bbox'], 
                    color=(0, 0, 0),
                    track_id=track_id,
                    label=piece['piece_type'],
                    distance=movement_data.get('total_distance', 0),
                    speed=movement_data.get('max_speed', 0)
                )

            output_video_frames.append(annotated_frame)

        return output_video_frames

    def _draw_piece_annotation(self, frame, bbox, color, track_id=None, label=None, distance=0, speed=0):
        """Draw a styled ellipse shadow and compact info badge per piece."""
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_bottom = int(bbox[3])
        width    = int(bbox[2] - bbox[0])
        is_white = (color != (0, 0, 0))

        # Shadow ellipse — chess.com green tint for white, muted for black
        ellipse_color = C_GREEN if is_white else C_MUTED
        cv2.ellipse(frame, (x_center, y_bottom),
                    (max(1, int(width * 0.55)), max(1, int(width * 0.18))),
                    0.0, 0, 360, ellipse_color, -1)
        cv2.ellipse(frame, (x_center, y_bottom),
                    (max(1, int(width * 0.55)), max(1, int(width * 0.18))),
                    0.0, 0, 360, C_PANEL_BG, 1)

        if track_id is None:
            return frame

        # Compact badge: piece symbol + ID on first line, dist on second
        sym   = PIECE_SYM.get(label, '?') if label else '?'
        bw, bh = 90, 36
        bx = x_center - bw // 2
        by = y_bottom + 4

        badge_bg     = C_PANEL_BG
        badge_border = C_GREEN if is_white else C_MUTED
        draw_rounded_rect(frame, (bx, by), (bx + bw, by + bh), badge_bg, radius=5)
        draw_rounded_rect(frame, (bx, by), (bx + bw, by + bh), badge_border, radius=5, thickness=1)

        def _draw(draw, _):
            top_col  = bgr_to_rgb(C_WHITE_IND if is_white else C_MUTED)
            bot_col  = bgr_to_rgb(C_MUTED)
            draw.text((bx + 5, by + 2),  f"{sym} #{track_id}", font=_font(12, bold=True), fill=top_col)
            draw.text((bx + 5, by + 20), f"{distance:.1f} sq", font=_font(10),            fill=bot_col)

        pil_draw(frame, _draw)
        return frame

class BoundingBoxStabilizer:
    def __init__(self, alpha=0.6):
        """
        Initialize stabilizer for bounding box coordinates.
        Args:
            alpha (float): Smoothing factor (0 < alpha <= 1).
        """
        self.alpha = alpha
        self.stabilized_bbox = None

    def update(self, new_bbox):
        """
        Update stabilized bounding box using exponential smoothing.
        Args:
            new_bbox (list): Current bounding box [x1, y1, x2, y2].
        Returns:
            list: Stabilized bounding box.
        """
        if self.stabilized_bbox is None:
            self.stabilized_bbox = new_bbox
        else:
            self.stabilized_bbox = [
                self.alpha * new + (1 - self.alpha) * old
                for new, old in zip(new_bbox, self.stabilized_bbox)
            ]
        return self.stabilized_bbox

def read_video(video_path, start_frame=0):
    """Read video frames, optionally skipping to start_frame."""
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}  |  Total frames: {total}  |  Start frame: {start_frame}")

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    print(f"Loaded {len(frames)} frames (from frame {start_frame} onwards)")
    return frames, fps

def _load_corrected_log(log_path):
    """
    Parse chessnut_moves_corrected.log and return:
      corrected_fens   : {abs_frame: fen_string, ...}
      corrected_moves  : sorted list of (abs_frame, uci_string)

    Line format (space-padded):
      abs_frame | method | uci | san | fen_after
    Lines starting with '#' or whitespace-only are skipped.
    """
    corrected_fens  = {}
    corrected_moves = []
    try:
        with open(log_path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 5:
                    continue
                try:
                    frame = int(parts[0])
                    uci   = parts[2]
                    fen   = parts[4]
                    corrected_fens[frame]  = fen
                    corrected_moves.append((frame, uci))
                except (ValueError, IndexError):
                    pass
    except FileNotFoundError:
        pass
    corrected_moves.sort()
    return corrected_fens, corrected_moves


def _best_fen_at(abs_frame, corrected_fens):
    """Return the FEN of the last corrected-log entry at or before abs_frame."""
    candidates = [f for f in corrected_fens if f <= abs_frame]
    if not candidates:
        return None
    return corrected_fens[max(candidates)]


def _gap_moves_in_window(ws, we, corrected_moves, already_pushed_upto):
    """
    Return UCI strings for corrected-log moves that fall inside [ws, we]
    and whose frame is > already_pushed_upto  (so we don't double-inject).
    """
    return [uci for (f, uci) in corrected_moves
            if ws <= f <= we and f > already_pushed_upto]


def main():
    # ── Input video ────────────────────────────────────────────────────────────
    video_path  = '/teamspace/studios/this_studio/chessnut_evo.mp4'
    START_FRAME = 115   # first frame where pieces start moving (~3.8 s in)

    if not os.path.exists(video_path):
        print("Error: Video file not found at", video_path)
        return

    # ── Output paths ───────────────────────────────────────────────────────────
    output_dir        = '/teamspace/studios/this_studio/chessnut_annotated_frames/'
    output_video_path = '/teamspace/studios/this_studio/chessnut_annotated.avi'
    pgn_path          = '/teamspace/studios/this_studio/chessnut_game.pgn'
    move_log_path     = '/teamspace/studios/this_studio/chessnut_moves.log'
    os.makedirs(output_dir, exist_ok=True)

    # ── Recording discontinuity handling ──────────────────────────────────────
    SKIP_WINDOWS_ABS = [
        (290,  338),    # short thinking gap (Nf3→Nc6); board unchanged
        (1416, 2015),   # long recording pause; board may have changed
    ]

    _skip_set   = set()
    _resume_set = set()
    for _ws, _we in SKIP_WINDOWS_ABS:
        for _f in range(_ws, _we + 1):
            _skip_set.add(_f - START_FRAME)
        _resume_set.add(_we + 1 - START_FRAME)

    _corrected_log = '/teamspace/studios/this_studio/chessnut_moves_corrected.log'
    _corrected_fens, _corrected_moves = _load_corrected_log(_corrected_log)
    if _corrected_fens:
        print(f"Loaded {len(_corrected_fens)} corrected FEN(s) from {_corrected_log}")
    else:
        print(f"No corrected log found at {_corrected_log} — FEN sync disabled")

    _last_injected_abs = 0

    # ── Move log: open for writing (overwrites previous run) ─────────────────
    move_log = open(move_log_path, 'w', buffering=1)   # line-buffered
    move_log.write('# Chessnut Evo — move log (YOLO11n-seg pipeline, chessnut_evo_ft_v1)\n')
    move_log.write('# Format: abs_frame | method | uci | san | fen_after\n')
    move_log.write('#' + '-' * 70 + '\n')

    # ── Probe video + grab the first frame — do NOT preload the whole video
    # into RAM (2680 frames × 1364×768×3 ≈ 8.4 GB was exhausting this box's
    # 15 GB RAM mid-run and getting the process silently OOM-killed). Frames
    # are streamed one at a time in the main loop below instead. ─────────────
    _probe_cap = cv2.VideoCapture(video_path)
    fps   = _probe_cap.get(cv2.CAP_PROP_FPS)
    total = int(_probe_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}  |  Total frames: {total}  |  Start frame: {START_FRAME}")
    if START_FRAME > 0:
        _probe_cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)
    _ret, first_frame = _probe_cap.read()
    _probe_cap.release()
    if not _ret or first_frame is None:
        print("Error: No frames were read from the video!")
        return

    # ── Models ─────────────────────────────────────────────────────────────────
    piece_model_path      = '/teamspace/studios/this_studio/runs/detect/finetune_v1/weights/best.pt'
    chessboard_model_path = '/teamspace/studios/this_studio/models/best.pt'
    seg_model_path         = '/teamspace/studios/this_studio/runs/segment/chessnut_evo_ft_v1/weights/best.pt'

    tracker = ChessTracker(
        model_path=piece_model_path,
        chessboard_model_path=chessboard_model_path,
        seg_model_path=seg_model_path
    )
    tracker.game_engine._fps = fps

    # ── Board orientation: white pieces appear at TOP of frame in this video ──
    tracker.square_mapper.white_at_bottom = False

    # ── Corner detection: both the inner-corner pattern detector and the LSD
    # detector fail outright on this board (tested across multiple clean,
    # hand-free frames with every flag combination — 0% success; the board's
    # white/light-grey squares are too low-contrast for calibration-pattern
    # algorithms, which expect a sharp black/white printed checkerboard).
    # prepare_tracking() still runs once below so its chain falls through to
    # the axis-aligned YOLO-bbox fallback and the debug viz gets written, but
    # we immediately overwrite that with precise corners computed from the
    # homography fit over 96 hand-labeled piece positions (chessnut_evo_seg_dataset
    # labeling, RMS ~6px) — this is the true perspective-correct quadrilateral
    # (near/bottom edge wider than the far/top edge), not a rectangle.
    #
    # IMPORTANT: the calibration anchor point must match SAM3Refiner.
    # mask_base_point exactly (single bottom-most mask row, not an average
    # over the lower 30% of the piece) — using a different anchor convention
    # shifted every piece's apparent square by about one row toward the
    # camera (e.g. the king on e1 was classified as e2), which silently broke
    # move detection: committed_state could never find a matching departure
    # square, so every move got absorbed into the per-square state without
    # ever being pushed to chess_board (engine stayed frozen at "Game start"
    # no matter how many real moves were played). Verified by re-running the
    # live seg model on frames 120-250 and checking classification against
    # every white back-rank piece's true home square (16/16 correct with
    # these corners vs 10/16 with the old ones). ────────────────────────────
    tracker.prepare_tracking(first_frame, headless=True)

    _locked_corners = np.array([
        [258.882, 118.393],   # TL — outer corner beyond h1
        [860.299, 114.709],   # TR — outer corner beyond a1
        [227.959, 703.390],   # BL — outer corner beyond h8
        [912.263, 692.027],   # BR — outer corner beyond a8
    ], dtype=np.float32)
    tracker.perspective_transformer.set_board_corners(_locked_corners)
    tracker.perspective_transformer.estimate_board_perspective()
    tracker._last_good_corners = _locked_corners.copy()
    print(f"Locked board corners for chessnut_evo.mp4 (perspective-fit, not axis-aligned): {_locked_corners.tolist()}")

    # ── Board guard (axis-aligned box clipped from the locked corners) ───────
    _board_guard = None
    if tracker.perspective_transformer.board_corners is not None:
        _bc = tracker.perspective_transformer.board_corners
        _board_guard = (int(_bc[:,0].min())-10, int(_bc[:,1].min())-10,
                        int(_bc[:,0].max())+10, int(_bc[:,1].max())+10)
        print(f"Board guard: {_board_guard}")

    # ── Output video writer ────────────────────────────────────────────────────
    video_writer = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (first_frame.shape[1], first_frame.shape[0])
    )

    # ── Initialise hand-occlusion detector ────────────────────────────────────
    occlusion_detector = OcclusionDetector(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    OUTLINE_COLOR = {
        BoardStateManager.S_STABLE:   (0, 255,   0),
        BoardStateManager.S_OCCLUDED: (0, 100, 255),
        BoardStateManager.S_SETTLING: (0, 220, 220),
    }

    # Two-colour mask palette matching the distribution panel
    _COL_WHITE_MASK = (130, 210, 255)   # warm amber (BGR)
    _COL_BLACK_MASK = (200, 100,  40)   # dark teal  (BGR)
    _seg_available  = tracker.seg_model is not None

    # Track-age gate: suppress detections until ByteTrack has confirmed them
    # for MIN_TRACK_HITS consecutive frames — filters single-frame YOLO false
    # positives that would otherwise corrupt BSM diffs.
    MIN_TRACK_HITS = 4
    _track_hits: dict = {}   # track_id → consecutive frames seen

    # Square-level stability for move detection.
    # We track stability per SQUARE (not per track ID) so that piece
    # pick-up/put-down across hand occlusion still registers as a move.
    # committed_squares: square → piece_type (last confirmed stable state)
    # square_run:        square → (piece_type, run_length)
    MOVE_STABLE_FRAMES  = 4     # frames a square must be consistently occupied/empty
    _sq_run:       dict = {}    # sq → (piece_type|None, run_length)
    # Seed _sq_committed from BSM starting position so first moves can be detected
    _sq_committed: dict = dict(tracker.board_state_manager.committed_state)
    # _sq_last_occupant never gets cleared to None on departure (unlike
    # _sq_committed) — it remembers what piece type last stably occupied a
    # square so a later arrival elsewhere can still be matched to it as the
    # departure square. Without this, a square that empties out and reaches
    # stability before the arrival square does (the common case — departure
    # and arrival rarely cross the stability threshold on the exact same
    # frame) has its committed entry overwritten to None, so the arrival can
    # never find it as a match and the move silently gets absorbed instead
    # of being pushed to chess_board.
    _sq_last_occupant: dict = dict(tracker.board_state_manager.committed_state)

    # Track-ID-based move detection (primary — see below for why this runs
    # before the square-based fallback). ByteTrack already gives each piece
    # a stable identity across frames (lost_track_buffer=90 survives a brief
    # hand occlusion during the actual move), so "this exact track's square
    # changed" is an unambiguous move signal — no risk of confusing which of
    # two same-type pieces (e.g. both knights) actually moved, which the
    # square-occupancy method below cannot distinguish on its own.
    _track_sq_run:       dict = {}   # tid → (square|None, run_length)
    _track_committed_sq: dict = {}   # tid → last confirmed-stable square
    _track_piece_type:   dict = {}   # tid → piece_type (most recently seen)

    # ── Stream frames one at a time from a fresh capture (no whole-video
    # preload — keeps memory flat regardless of video length) ────────────────
    _video_cap = cv2.VideoCapture(video_path)
    if START_FRAME > 0:
        _video_cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    frame_num = -1
    while True:
        frame_num += 1
        _ret_f, frame = _video_cap.read()
        if not _ret_f:
            break   # end of video
        if frame is None:
            print(f"Warning: Frame {frame_num} is None, skipping...")
            continue

        # ── Step 1: detect + segment in one pass (falls back to detect-only
        # if the seg model wasn't loaded) ──────────────────────────────────────
        if _seg_available:
            _result = tracker.seg_model.predict(frame, conf=0.25, verbose=False)[0]
        else:
            _result = tracker.model.predict(frame, conf=0.25, verbose=False)[0]
        _det_sv = sv.Detections.from_ultralytics(_result)
        _det_sv = _cross_class_nms(_det_sv, iou_threshold=0.5)

        # ── Step 2: ByteTrack — stabilise positions ───────────────────────────
        _tracked = tracker.tracker.update_with_detections(_det_sv)

        # Update hit counters — increment active tracks, age out lost ones
        _active_ids = set(int(t) for t in _tracked.tracker_id)
        for _tid in _active_ids:
            _track_hits[_tid] = _track_hits.get(_tid, 0) + 1
        for _tid in list(_track_hits):
            if _tid not in _active_ids:
                _track_hits[_tid] = max(0, _track_hits[_tid] - 1)
                if _track_hits[_tid] == 0:
                    del _track_hits[_tid]

        # ── Step 3: masks from the seg model's own output ─────────────────────
        # chessnut_evo_ft_v1 is a detect+segment model, so Step 1 already
        # produced per-detection masks — no separate SAM2 refine pass needed.
        # Match ByteTrack's (possibly reordered/filtered) boxes back to the
        # pre-tracking detection rows by nearest centre, same as the class-id
        # lookup below, and pull each tracked box's mask from that row.
        _masks_np   = None   # (N, H, W) float numpy
        _cls_ids_tv = []     # class ids aligned with _tracked rows

        if _seg_available and len(_tracked.xyxy) > 0 and _result.masks is not None:
            # Re-derive class IDs from the original YOLO result for each tracked
            # box by matching track bbox centres to detection bbox centres —
            # always computed (even if mask matching below fails) so Step 4
            # still gets correct piece types.
            _det_centres = ((_result.boxes.xyxy[:,0]+_result.boxes.xyxy[:,2])/2,
                            (_result.boxes.xyxy[:,1]+_result.boxes.xyxy[:,3])/2)
            _det_cls  = _result.boxes.cls.int().cpu().numpy()
            _match_idx = []
            for _tb in _tracked.xyxy:
                _tcx, _tcy = (_tb[0]+_tb[2])/2, (_tb[1]+_tb[3])/2
                _dists = ((_det_centres[0]-_tcx)**2 + (_det_centres[1]-_tcy)**2)
                _mi = int(_dists.argmin())
                _match_idx.append(_mi)
                _cls_ids_tv.append(int(_det_cls[_mi]))

            try:
                _H, _W = frame.shape[:2]
                _det_mask = _result.masks.data.cpu().numpy()   # (M,h,w) proto resolution
                _matched_masks = []
                for _mi in _match_idx:
                    _m = _det_mask[_mi]
                    if _m.shape[:2] != (_H, _W):
                        _m = cv2.resize(_m, (_W, _H), interpolation=cv2.INTER_NEAREST)
                    _matched_masks.append(_m)
                _masks_np = np.stack(_matched_masks, axis=0)

                # Refine bboxes from mask extents (tight fit around silhouette)
                _refined = []
                for _i, _m in enumerate(_masks_np):
                    _yx = np.argwhere(_m > 0.5)
                    if len(_yx):
                        _y1,_x1 = _yx.min(0); _y2,_x2 = _yx.max(0)
                        _refined.append([
                            max(0,_x1-3), max(0,_y1-3),
                            min(_W,_x2+3), min(_H,_y2+3)
                        ])
                    else:
                        _refined.append(_tracked.xyxy[_i].tolist())
                _tracked.xyxy = np.array(_refined, dtype=np.float32)
            except Exception as _e:
                print(f"  seg mask match error frame {frame_num}: {_e}")
                _masks_np = None

        # ── Step 4: Board guard + build frame-track dicts ─────────────────────
        _frame_white: dict = {}
        _frame_black: dict = {}
        _cls_names = _result.names

        for _i, (_bbox, _tid) in enumerate(zip(_tracked.xyxy, _tracked.tracker_id)):
            _bbox  = _bbox.tolist()
            _tid   = int(_tid)

            # Track-age gate: ignore immature tracks (likely false positives)
            if _track_hits.get(_tid, 0) < MIN_TRACK_HITS:
                continue

            _cid   = _cls_ids_tv[_i] if _i < len(_cls_ids_tv) else 0
            _pname = _cls_names[_cid]

            if _board_guard is not None:
                _cx = (_bbox[0]+_bbox[2])/2; _cy = (_bbox[1]+_bbox[3])/2
                _gx1,_gy1,_gx2,_gy2 = _board_guard
                if not (_gx1<=_cx<=_gx2 and _gy1<=_cy<=_gy2):
                    continue

            # Base-point: bottom-centre of SAM2 mask if available, else bbox
            if _masks_np is not None and _i < len(_masks_np):
                _bp = SAM3Refiner.mask_base_point(_masks_np[_i])
                _base_pt = _bp if _bp else (int((_bbox[0]+_bbox[2])/2), int(_bbox[3]))
            else:
                _base_pt = (int((_bbox[0]+_bbox[2])/2), int(_bbox[3]))

            _pos   = ((_bbox[0]+_bbox[2])/2, (_bbox[1]+_bbox[3])/2)
            _entry = {'bbox': _bbox, 'piece_type': _pname,
                      'conf': 0.5, 'position': _pos, 'base_pt': _base_pt}

            if _pname.startswith('white'):
                _frame_white[_tid] = _entry
            elif _pname.startswith('black'):
                _frame_black[_tid] = _entry

        # Royalty constraints (king/queen disambiguation)
        _frame_white = tracker._enforce_royalty_constraints(
            _frame_white, 'white', frame=frame)
        _frame_black = tracker._enforce_royalty_constraints(
            _frame_black, 'black', frame=frame)

        # ── Hand detection ────────────────────────────────────────────────────
        board_corners = tracker.perspective_transformer.board_corners
        hand_present, hand_boxes, hand_landmarks = occlusion_detector.detect(
            frame, board_corners)

        # ── Board outline + grid (drawn on frame before copy so always visible) ─
        if board_corners is not None:
            corners     = board_corners.astype(int)
            state       = tracker.board_state_manager.occlusion_state
            outline_col = OUTLINE_COLOR.get(state, (0, 255, 0))
            cv2.polylines(frame, [corners[[0, 1, 3, 2]]], True, outline_col, 2)
            tracker.square_mapper.draw_square_grid(frame)

        annotated_frame = frame.copy()

        # ── Step 5: Draw seg masks (two-colour uniform scheme) ────────────────
        if _masks_np is not None:
            _overlay = annotated_frame.copy()
            for _i, (_m, _cid) in enumerate(zip(_masks_np, _cls_ids_tv)):
                _pn  = _cls_names[_cid]
                _col = _COL_WHITE_MASK if _pn.startswith('white') else _COL_BLACK_MASK
                _bin = (_m > 0.5).astype(np.uint8)
                if (int(_H),int(_W)) != _bin.shape:
                    _bin = cv2.resize(_bin, (int(_W),int(_H)), cv2.INTER_NEAREST)
                _cnts,_ = cv2.findContours(_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.fillPoly(_overlay, _cnts, _col)
                cv2.drawContours(annotated_frame, _cnts, -1, _col, 2)
            cv2.addWeighted(_overlay, 0.35, annotated_frame, 0.65, 0, annotated_frame)

        # ── Hand overlay ──────────────────────────────────────────────────────
        if hand_present:
            for (x1, y1, x2, y2) in hand_boxes:
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2),
                              (0, 100, 255), 2)
            for pts in hand_landmarks:
                for (a, b) in _HAND_CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(annotated_frame, pts[a], pts[b],
                                 (0, 200, 255), 1, cv2.LINE_AA)
                for i, (kx, ky) in enumerate(pts):
                    r = 5 if i in _FINGERTIPS else 3
                    cv2.circle(annotated_frame, (kx, ky), r,
                               (0, 80, 255), -1, cv2.LINE_AA)
                    cv2.circle(annotated_frame, (kx, ky), r,
                               (210, 235, 255), 1, cv2.LINE_AA)

        # Highlight legal moves for white pieces
        for _tid, piece in _frame_white.items():
            annotated_frame = tracker.highlight_possible_moves(
                annotated_frame, piece['position'],
                tracker.board_state_manager.chess_board
            )

        # ── Step 6: Square mapping using mask base-points ─────────────────────
        # Build a minimal tracks structure and override base-point lookup
        _all_pieces = ([(track_id, info, 'white') for track_id, info in _frame_white.items()] +
                       [(track_id, info, 'black') for track_id, info in _frame_black.items()])

        board_state: dict = {}
        _conflicts:  dict = {}
        for _tid, _info, _side in _all_pieces:
            _bx, _by = _info['base_pt']
            _sq = tracker.square_mapper.pixel_to_square(_bx, _by)
            if _sq is None:
                continue
            if _sq not in _conflicts or _by > _conflicts[_sq]:
                board_state[_sq] = _info['piece_type']
                _conflicts[_sq]  = _by

        # ── Bounding box visualization ────────────────────────────────────────
        for _tid, _info, _side in _all_pieces:
            _bx1, _by1, _bx2, _by2 = [int(v) for v in _info['bbox']]
            _color = (0, 255, 0) if _side == 'white' else (255, 0, 0)
            cv2.rectangle(annotated_frame, (_bx1, _by1), (_bx2, _by2), _color, 2)
            _sq = tracker.square_mapper.pixel_to_square(
                int((_bx1 + _bx2) / 2), _by2)
            _label = f"{_info['piece_type'].split('-')[1][0].upper()}{_sq or '?'}"
            cv2.putText(annotated_frame, _label, (_bx1, _by1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _color, 1)

        # ── Recording-gap handling ─────────────────────────────────────────────
        abs_frame = START_FRAME + frame_num   # absolute frame number

        if frame_num in _resume_set:
            # First frame after a recording pause.
            #
            # Two cases:
            #   SHORT gap (≤ 50 frames): only thinking time was cut; the board
            #     state hasn't changed.  Don't resync — let the BSM diff from the
            #     pre-gap committed_state and detect any move (e.g. Nc6) normally.
            #   LONG gap (> 50 frames): multiple moves were likely made while the
            #     recording was paused.  Reset committed_state to the current YOLO
            #     observation so BFS doesn't try to reconcile an impossible diff.
            _ws_abs = next((ws for (ws, we) in SKIP_WINDOWS_ABS
                            if we + 1 - START_FRAME == frame_num), None)
            _we_abs = next((we for (ws, we) in SKIP_WINDOWS_ABS
                            if ws == _ws_abs), None) if _ws_abs is not None else None
            _gap_len = (_we_abs - _ws_abs) if (_ws_abs is not None and _we_abs is not None) else 0

            _GAP_SHORT_THRESHOLD = 50   # frames; below this = thinking-only pause

            if _gap_len > _GAP_SHORT_THRESHOLD:
                # Long gap — resync committed_state + optionally fix chess_board FEN.
                _gap_ucis  = _gap_moves_in_window(
                    _ws_abs, _we_abs, _corrected_moves, _last_injected_abs)

                # Only use a corrected-log FEN to reset chess_board if it is
                # "recent" — i.e. the log entry is within 200 frames of the
                # resume point.  A stale FEN (e.g. from 1500 frames ago) would
                # rewind chess_board further back than it already is, corrupting
                # turn tracking and producing wrong SANs for subsequent moves.
                _FEN_STALENESS_LIMIT = 200
                _best_fen_frame = max(
                    (f for f in _corrected_fens if f <= abs_frame), default=None)
                _resync_fen = (
                    _corrected_fens[_best_fen_frame]
                    if _best_fen_frame is not None
                       and (abs_frame - _best_fen_frame) <= _FEN_STALENESS_LIMIT
                    else None
                )
                print(f"\n[DISCONTINUITY-LONG] abs_frame={abs_frame}  "
                      f"gap={_gap_len}f  gap_moves={_gap_ucis}  "
                      f"fen={'set (frame %d)' % _best_fen_frame if _resync_fen else 'skipped (stale)'}")
                tracker.board_state_manager.force_resync(
                    board_state,
                    chess_fen=_resync_fen,
                    gap_moves_uci=_gap_ucis,
                )
                _last_injected_abs = abs_frame
                # update() will see ~0 diff → no spurious move.
            else:
                # Short gap — board hasn't changed; let BSM detect moves normally.
                print(f"\n[DISCONTINUITY-SHORT] abs_frame={abs_frame}  "
                      f"gap={_gap_len}f  resuming normal detection")

        elif frame_num in _skip_set:
            # Inside a recording gap.  The board state is unknown (moves may
            # have been made off-camera).  Suppress BSM update entirely so no
            # phantom move is inferred from the garbage diff.
            detected_move, det_method = None, ''
            # Still do visual rendering below; just skip the move-log write.
            # Same engine-authoritative counts as the main path, for consistency.
            _skip_engine_state = tracker.board_state_manager._state_from_chess_board(
                tracker.board_state_manager.chess_board)
            _skip_white = sum(1 for p in _skip_engine_state.values() if 'white' in p) if _skip_engine_state else 0
            _skip_black = sum(1 for p in _skip_engine_state.values() if 'black' in p) if _skip_engine_state else 0
            annotated_frame = tracker.draw_statistics_overlay(annotated_frame, _skip_white, _skip_black)
            annotated_frame = tracker.square_mapper.draw_piece_assignments(
                annotated_frame, board_state)
            annotated_frame = tracker.board_state_manager.draw_game_overlay(
                annotated_frame)
            annotated_frame = tracker.game_engine.annotate_frame(
                annotated_frame, tracker.square_mapper,
                detected_board_state=board_state)
            abs_frame_vis = START_FRAME + frame_num
            output_frame_path = os.path.join(
                output_dir, f"frame_{abs_frame_vis:04d}.jpg")
            cv2.imwrite(output_frame_path, annotated_frame)
            video_writer.write(annotated_frame)
            continue   # skip the rest of the loop body

        # ── Per-square / per-track move detection ──────────────────────────────
        # occupied_now: squares seen in this frame (also drives the per-track
        # square history used by the Tier-1 detector below)
        detected_move, det_method = None, ''
        _bsm = tracker.board_state_manager
        _occupied_now = {}       # sq → piece_type for this frame
        _track_now_sq = {}       # tid → sq for this frame
        for _tid, _info, _side in _all_pieces:
            _bx, _by = _info['base_pt']
            _sq = tracker.square_mapper.pixel_to_square(_bx, _by)
            if _sq is None:
                continue
            _occupied_now[_sq] = _info['piece_type']
            _track_now_sq[_tid] = _sq
            _track_piece_type[_tid] = _info['piece_type']

        # Update run lengths for all squares (occupied and empty)
        _all_squares = set(_sq_run) | set(_occupied_now) | set(_sq_committed)
        for _sq in _all_squares:
            _ptype_now = _occupied_now.get(_sq)   # None if empty this frame
            _prev_ptype, _prev_run = _sq_run.get(_sq, (None, 0))
            if _ptype_now == _prev_ptype:
                _sq_run[_sq] = (_ptype_now, _prev_run + 1)
            else:
                _sq_run[_sq] = (_ptype_now, 1)

        # Update run lengths for all tracks seen so far (per track id)
        _all_tracks = set(_track_sq_run) | set(_track_now_sq)
        for _tid in _all_tracks:
            _tsq_now = _track_now_sq.get(_tid)   # None if not detected this frame
            _prev_tsq, _prev_trun = _track_sq_run.get(_tid, (None, 0))
            if _tsq_now == _prev_tsq:
                _track_sq_run[_tid] = (_tsq_now, _prev_trun + 1)
            else:
                _track_sq_run[_tid] = (_tsq_now, 1)

        # ── Tier 1: track-identity-based move detection ────────────────────────
        # A single tracked piece stably changing which square it occupies is an
        # unambiguous move — no guessing which of several same-type pieces it
        # was. Runs before the square-occupancy fallback so it gets first claim
        # on the one-move-per-frame slot.
        for _tid, (_tsq, _trun) in list(_track_sq_run.items()):
            if _tsq is None or _trun < MOVE_STABLE_FRAMES:
                continue
            _prev_tsq = _track_committed_sq.get(_tid)
            if _prev_tsq is None:
                # First time this track has stabilized anywhere (initial
                # position, or a fresh ID after occlusion broke continuity) —
                # record it; no move to infer from a single data point.
                _track_committed_sq[_tid] = _tsq
                continue
            if _prev_tsq == _tsq:
                continue   # already where we last committed it
            if detected_move is not None:
                continue   # one move per frame — retry next frame
            _ptype = _track_piece_type.get(_tid)
            try:
                _move = chess.Move.from_uci(f"{_prev_tsq}{_tsq}")
                if _move in _bsm.chess_board.legal_moves:
                    detected_move = _move
                    det_method = 'track'
                    _track_committed_sq[_tid] = _tsq
                    _sq_committed[_prev_tsq] = None
                    _sq_committed[_tsq] = _ptype
                    _sq_last_occupant[_tsq] = _ptype
                    print(f"Move detected [track]: {_prev_tsq}→{_tsq} "
                          f"({_ptype}) tid={_tid}  frame={abs_frame}", flush=True)
                else:
                    # Square changed but isn't a legal move from there — this
                    # track's identity likely jumped to a different physical
                    # piece (ByteTrack ID reuse). Resync without claiming a
                    # move; the square-occupancy fallback may still catch the
                    # real move via _sq_last_occupant.
                    _track_committed_sq[_tid] = _tsq
            except Exception:
                _track_committed_sq[_tid] = _tsq

        # ── Tier 2: square-occupancy fallback (for when track identity breaks,
        # e.g. a long hand occlusion re-assigns a new ByteTrack id) ───────────
        for _sq, (_ptype_now, _run) in list(_sq_run.items()):
            if _run < MOVE_STABLE_FRAMES:
                continue
            _prev_committed = _sq_committed.get(_sq)

            if _ptype_now is not None:
                _sq_last_occupant[_sq] = _ptype_now

            if _ptype_now is not None and _ptype_now != _prev_committed:
                # Square stably occupied by a new piece — find departure square.
                # Match against _sq_last_occupant (never cleared on departure),
                # not _sq_committed (which is already None for squares that
                # emptied out before this arrival reached stability).
                if detected_move is None:   # only one move per frame
                    _candidates = [
                        s for s, p in _sq_last_occupant.items()
                        if p == _ptype_now and s != _sq
                        and _sq_run.get(s, (None, 0))[0] is None
                        and _sq_run.get(s, (None, 0))[1] >= MOVE_STABLE_FRAMES
                    ]
                    # Nearest first — when several same-type pieces have left
                    # their square (e.g. both knights developed before track
                    # identity broke), a real chess move is almost always the
                    # geometrically closest match, so try that before giving up
                    # on farther, less plausible candidates.
                    _candidates.sort(key=lambda s: _square_dist(s, _sq))
                    for _from_sq in _candidates:
                        try:
                            _move = chess.Move.from_uci(f"{_from_sq}{_sq}")
                            if _move in _bsm.chess_board.legal_moves:
                                detected_move = _move
                                det_method = 'square'
                                # Update committed for both squares immediately
                                _sq_committed[_from_sq] = None
                                _sq_committed[_sq] = _ptype_now
                                print(f"Move detected [square]: {_from_sq}→{_sq} "
                                      f"({_ptype_now})  frame={abs_frame}", flush=True)
                                break
                        except Exception:
                            continue
                    if detected_move is not None:
                        continue  # skip the committed update below
                _sq_committed[_sq] = _ptype_now

            elif _ptype_now is None and _prev_committed is not None:
                # Square stably empty — piece departed; update committed
                _sq_committed[_sq] = None
        if detected_move:
            bsm = tracker.board_state_manager
            # Push move to chess_board and update all tracking state
            try:
                bsm.chess_board.push(detected_move)
                bsm.move_history.append(detected_move)
                bsm.move_methods.append('square')
                bsm.last_move = detected_move
                # Pre-populate GameEngine SAN cache so _rebuild_san_cache
                # exits early without destructive replay or truncation
                ge = tracker.game_engine
                ge._san_cache.append(detected_move.uci())
                ge._san_cache_len = len(bsm.move_history)
                # Set arrow TTL so the move arrow is drawn
                ge._last_move_frame = frame_num
                ge._ghost_arrow_ttl = 30
                ge._move_annotations.append(('', (200, 200, 200)))
                # Write move log
                san = bsm.get_last_move_san()
                fen_after = bsm.chess_board.fen()
                move_log.write(
                    f"{abs_frame:5d} | {det_method:9s} | {detected_move.uci():6s} "
                    f"| {san:8s} | {fen_after}\n"
                )
                move_log.flush()
                print(f"Move committed: {detected_move.uci()} SAN={san} frame={abs_frame}", flush=True)
                if len(bsm.move_history) % 50 == 0:
                    tracker.game_engine.save_pgn(pgn_path)
            except Exception as _push_exc:
                print(f"[warn] push failed: {_push_exc}", flush=True)
                detected_move = None
        # ── Visual board state: augment YOLO detections with engine knowledge ──
        # Draw ALL pieces the chess engine knows are on the board, even if YOLO
        # missed them this frame. This keeps annotations complete while the BSM
        # still receives only raw YOLO detections (no diff pollution).
        _bsm = tracker.board_state_manager
        board_state_visual = dict(board_state)
        _engine_state = _bsm._state_from_chess_board(_bsm.chess_board)
        if _engine_state:
            for _sq, _piece in _engine_state.items():
                if _sq not in board_state_visual:
                    board_state_visual[_sq] = _piece

        # Piece-count panel uses the engine's authoritative state, not raw
        # per-frame detections — the engine always has all 32 pieces at the
        # start and only loses them on confirmed captures, so counts stay
        # correct even on frames where the seg model misses a piece.
        _white_count = sum(1 for p in _engine_state.values() if 'white' in p) if _engine_state else 0
        _black_count = sum(1 for p in _engine_state.values() if 'black' in p) if _engine_state else 0
        annotated_frame = tracker.draw_statistics_overlay(annotated_frame, _white_count, _black_count)

        annotated_frame = tracker.square_mapper.draw_piece_assignments(annotated_frame, board_state_visual)
        annotated_frame = tracker.board_state_manager.draw_game_overlay(annotated_frame)
        annotated_frame = tracker.game_engine.annotate_frame(
            annotated_frame, tracker.square_mapper,
            # board_state_visual (engine-filled gaps), not raw board_state —
            # otherwise the distribution panel drops any piece the seg model
            # missed in this one frame, same class of bug as the piece-count fix.
            detected_board_state=board_state_visual
        )

        # Save annotated frame (number relative to START_FRAME for clarity)
        output_frame_path = os.path.join(output_dir, f"frame_{abs_frame:04d}.jpg")
        cv2.imwrite(output_frame_path, annotated_frame)

        # Write to video
        video_writer.write(annotated_frame)

        # ── Periodic PGN autosave every 200 frames ───────────────────────────
        if frame_num % 200 == 0 and frame_num > 0:
            tracker.game_engine.save_pgn(pgn_path)

        # Log board state only for frames where a move was detected (reduces verbosity)
        if board_state and detected_move:
            print(f"\nFrame {abs_frame} (rel {frame_num}) board state after move:")
            for square in sorted(board_state):
                print(f"  {square}: {board_state[square]}")

    # Finalise
    _video_cap.release()
    video_writer.release()
    occlusion_detector.close()
    move_log.write('#' + '-' * 70 + '\n')
    move_log.write(f'# Total moves logged: {len(tracker.board_state_manager.move_history)}\n')
    move_log.close()

    # Save PGN and shut down analysis engine
    tracker.game_engine.save_pgn(pgn_path)
    tracker.game_engine.close()

    print(f"\nAnnotated frames saved to: {output_dir}")
    print(f"Annotated video saved to: {output_video_path}")
    print(f"PGN saved to: {pgn_path}")


if __name__ == '__main__':
    main()