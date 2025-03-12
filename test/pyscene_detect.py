from scenedetect import open_video, SceneManager, ContentDetector
from scenedetect.detectors import AdaptiveDetector


# 打开视频文件
prefix = "/Users/user/Downloads/anime/JoJo_Part_6_Stone_Ocean_1_24_BDRip_1080p_HEVC_DTS_RAW"
name = "_Anime_Land__JoJo_Stone_Ocean_19__BDRip_1080p_HEVC_DTS___F6167DCD.mkv"
name = "_Anime_Land__JoJo_Stone_Ocean_22__BDRip_1080p_HEVC_DTS___20184B91.mkv"
video_path = f'{prefix}/{name}'  # 替换为你的视频文件路径
video = open_video(video_path)

# 创建 SceneManager 并添加检测器
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=27.0, min_scene_len=15))
scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3.0, min_scene_len=15))

# 检测视频中的场景
scene_manager.detect_scenes(video)

# 获取场景列表
scene_list = scene_manager.get_scene_list()
for i, scene in enumerate(scene_list):
    print(f"场景 {i+1}: 开始时间 {scene[0].get_timecode()} / 帧数 {scene[0].get_frames()}, "
          f"结束时间 {scene[1].get_timecode()} / 帧数 {scene[1].get_frames()}")
