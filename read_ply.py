from pyntcloud import PyntCloud


human_face = PyntCloud.from_file("reconstructed.ply")
human_face.plot()