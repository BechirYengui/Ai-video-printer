"""YVP : génère des photos de profil LinkedIn avec PhotoMaker V2 (SDXL).

Lancé par app/services/headshot.py dans l'environnement .headshot-venv
(PyTorch CUDA), jamais importé par l'application. Tout passe par des fichiers
du dossier de la tâche :
- job.json    (entrée)  : photos, prompt, nombre d'images, taille, étapes, graine ;
- status.json (sortie)  : phase, progression, photos écartées, images produites ;
- headshot-N.png        : les résultats.

Usage : python worker.py <dossier_de_la_tâche>   (ou --check : vérifie l'installation)
"""

import gc
import json
import os
import shutil
import sys
import time
import traceback

job_dir = sys.argv[1] if len(sys.argv) > 1 else ""
status = {"status": "running", "phase": "analyze", "fraction": 0.0, "started": time.time(), "rejected": [], "outputs": [], "error": ""}


def write_status(**fields):
    status.update(fields)
    tmp = os.path.join(job_dir, "status.json.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False)
    os.replace(tmp, os.path.join(job_dir, "status.json"))


def face_issue(faces):
    """Raison d'écarter une photo, ou "" si son visage principal est exploitable."""
    if not faces:
        return "no_face"
    areas = sorted(((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]) for f in faces), reverse=True)
    # Un second visage presque aussi grand : l'identité serait mélangée.
    if len(areas) > 1 and areas[1] > 0.3 * areas[0]:
        return "several_faces"
    if areas[0] < 90 * 90:
        return "face_too_small"
    return ""


def face_crop(image, bbox, scale=2.0):
    """Carré centré sur le visage (2× sa taille) : le module d'identité réduit
    chaque photo à 224 px, un selfie entier y laisse un visage de ~80 px.
    Mesuré sur des selfies réels : +0,03 de ressemblance (ArcFace)."""
    x0, y0, x1, y1 = bbox
    side = min(scale * max(x1 - x0, y1 - y0), image.width, image.height)
    left = min(max((x0 + x1) / 2 - side / 2, 0), image.width - side)
    top = min(max((y0 + y1) / 2 - side / 2, 0), image.height - side)
    return image.crop((int(left), int(top), int(left + side), int(top + side)))


def merge_step(steps, ratio=0.15):
    """Étape où l'identité entre dans l'image. La démo officielle prend 20 % des
    étapes ; 15 % ressemble un peu plus (mesuré : 0,56 → 0,60 avec le recadrage)."""
    return max(1, round(ratio * steps))


def analyze(job, max_photos):
    import numpy as np
    import torch
    from diffusers.utils import load_image
    from faces import FaceAnalyzer

    faces_app = FaceAnalyzer()
    kept, rejected = [], []
    photos = job["photos"]
    for index, name in enumerate(photos):
        image = load_image(os.path.join(job_dir, "photos", name))
        faces = faces_app.get(np.array(image)[:, :, ::-1])
        issue = face_issue(faces)
        if not issue:
            face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            if face.det_score < 0.6:
                issue = "unclear_face"
        if issue:
            rejected.append({"name": job.get("labels", {}).get(name, name), "reason": issue})
        else:
            size = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
            kept.append((size * float(face.det_score), face_crop(image, face.bbox),
                         torch.from_numpy(face.embedding)))
        write_status(fraction=(index + 1) / len(photos), rejected=rejected)
    # Les visages les plus grands et les plus nets d'abord ; au-delà de
    # max_photos, la ressemblance ne gagne plus rien et la mémoire GPU déborde.
    kept.sort(key=lambda item: item[0], reverse=True)
    kept = kept[:max_photos]
    write_status(used=len(kept))
    return [item[1] for item in kept], [item[2] for item in kept]


def model_path(repo, patterns):
    """Le cache local d'abord : la connexion est lente et coupe souvent."""
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo, allow_patterns=patterns, local_files_only=True)
    except Exception:
        pass
    last_error = None
    for _ in range(20):
        try:
            return snapshot_download(repo, allow_patterns=patterns, max_workers=2)
        except Exception as exc:  # coupure réseau : on reprend là où on en était
            last_error = exc
            time.sleep(10)
    raise RuntimeError(f"téléchargement impossible : {repo} ({last_error})")


def release_ram():
    """Rend au système la RAM libérée par Python et PyTorch (sinon glibc la garde)."""
    gc.collect()
    if sys.platform.startswith("linux"):
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)


def stub_insightface():
    """photomaker importe insightface pour un outil qu'on n'utilise pas (faces.py le
    remplace). Sous Windows il n'est pas installé : un module vide suffit à l'import."""
    try:
        import insightface  # noqa: F401
    except ImportError:
        import types

        for name in ("insightface", "insightface.app", "insightface.data"):
            sys.modules[name] = types.ModuleType(name)
        sys.modules["insightface.app"].FaceAnalysis = object
        sys.modules["insightface.data"].get_image = None


def load_pipeline(job):
    import torch
    from diffusers import EulerDiscreteScheduler

    stub_insightface()
    from photomaker import PhotoMakerStableDiffusionXLPipeline

    base = model_path(job["base_model"], [
        "model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
        "text_encoder/config.json", "text_encoder/model.fp16.safetensors",
        "text_encoder_2/config.json", "text_encoder_2/model.fp16.safetensors",
        "unet/config.json", "unet/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json", "vae/diffusion_pytorch_model.fp16.safetensors",
    ])
    adapter = model_path("TencentARC/PhotoMaker-V2", ["photomaker-v2.bin"])
    pipe = PhotoMakerStableDiffusionXLPipeline.from_pretrained(
        base, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    )
    pipe.load_photomaker_adapter(adapter, subfolder="", weight_name="photomaker-v2.bin",
                                 trigger_word="img", pm_version="v2")
    pipe.fuse_lora()
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    # Carte de 8 Go : les sous-modèles passent sur la carte à tour de rôle.
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    # Le module d'identité n'est pas géré par le déchargement : il reste sur la carte.
    pipe.id_encoder.to("cuda")
    return pipe


def main():
    with open(os.path.join(job_dir, "job.json"), encoding="utf-8") as handle:
        job = json.load(handle)
    write_status()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("no_cuda")
    images, embeds = analyze(job, job.get("max_photos", 10))
    if len(images) < job.get("min_usable", 3):
        write_status(status="failed", error="not_enough_faces")
        return

    write_status(phase="load", fraction=0.0)
    pipe = load_pipeline(job)
    release_ram()

    count, steps = job["count"], job["steps"]
    generator = torch.Generator(device="cpu").manual_seed(job["seed"])
    latents = []
    for index in range(count):
        write_status(phase="generate", current=index + 1, total=count, fraction=index / count)

        def on_step(_pipe, step, _timestep, kwargs, index=index):
            write_status(fraction=(index + (step + 1) / steps) / count)
            return kwargs

        # Sortie « latent » : le pipeline rend la main sans renvoyer l'UNet en RAM.
        latents.append(pipe(
            prompt=job["prompt"], negative_prompt=job["negative_prompt"],
            input_id_images=images, id_embeds=torch.stack(embeds),
            num_inference_steps=steps, start_merge_step=merge_step(steps),
            guidance_scale=job.get("guidance_scale", 5.0),
            height=job["size"], width=job["size"], generator=generator,
            callback_on_step_end=on_step, output_type="latent",
        ).images)
        if index < count - 1:
            # Image suivante : place aux encodeurs de texte sur la carte.
            pipe.maybe_free_model_hooks()
            release_ram()

    # Décodage sans l'UNet ni les encodeurs : avant, l'UNet revenait en RAM pour
    # laisser la carte au VAE (mesuré : 6 → 8,2 Go), et ce pic faisait tuer YVP par
    # systemd-oomd. Ici on le jette directement depuis la carte.
    vae, processor = pipe.vae, pipe.image_processor
    pipe.remove_all_hooks()
    del pipe
    release_ram()
    torch.cuda.empty_cache()
    outputs = []
    for index, image in enumerate(decode(vae, processor, latents)):
        name = f"headshot-{index + 1}.png"
        image.save(os.path.join(job_dir, name))
        outputs.append(name)
        write_status(outputs=outputs)
    write_status(status="done", fraction=1.0, finished=time.time())


def decode(vae, processor, latents):
    """Fin du pipeline SDXL (VAE en float32 : il déborde en float16), image par image."""
    import torch

    dtype = torch.float32 if vae.config.force_upcast else vae.dtype
    vae.to("cuda", dtype=dtype)
    for batch in latents:
        with torch.no_grad():
            pixels = vae.decode(batch.to("cuda", dtype) / vae.config.scaling_factor, return_dict=False)[0]
        yield processor.postprocess(pixels, output_type="pil")[0]


def check():
    """Utilisé par setup.sh / setup.ps1 : tout ce dont la génération a besoin s'importe."""
    import cv2, diffusers, onnxruntime, torch  # noqa: F401
    import faces  # noqa: F401

    stub_insightface()
    import photomaker  # noqa: F401
    print("imports OK")


if __name__ == "__main__" and job_dir == "--check":
    check()
    sys.exit(0)

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        message = str(exc)
        if "out of memory" in message.lower():
            message = "out_of_memory"
        write_status(status="failed", error=message[:500])
        sys.exit(1)
    finally:
        # Les photos du visage ne restent pas sur le disque après la génération.
        shutil.rmtree(os.path.join(job_dir, "photos"), ignore_errors=True)
