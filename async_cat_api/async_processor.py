import os
import asyncio
import time
import aiohttp
import aiofiles
import cv2
from typing import List, Dict, AsyncGenerator
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from cat_api.cat_image import CatImage, ColorCatImage
from cat_api.processor import CatImageProcessor

def _cpu_bound_edge_detect(img_bytes: bytes, url: str, breed: str) -> "CatImage":
    """
    Функция для выполнения в отдельном процессе.
    Десериализует изображение и выполняет свертку.
    """
    pid = os.getpid()
    print(f"Convolution for image {breed} started (PID {pid})")
    
    cat_img = CatImage.from_bytes(img_bytes, url, breed)
    
    edges_lib = cat_img.edge_detect_library()
    edges_custom = cat_img.edge_detect_custom()
    
    print(f"Convolution for image {breed} finished (PID {pid})")
    return cat_img, edges_lib, edges_custom

class AsyncCatImageProcessor(CatImageProcessor):
    def __init__(self, provider: str = "cat", output_dir: str = "downloads", limit: int = 1):
        super().__init__(provider, output_dir, limit)

    async def fetch_json(self, session: aiohttp.ClientSession) -> List[Dict]:
        url = self._base_url()
        params = {
            "size": "med",
            "mime_types": "jpg,png",
            "format": "json",
            "has_breeds": "true",
            "order": "RANDOM",
            "page": "0",
            "limit": str(self.limit),
        }
        headers = {"x-api-key": self._api_key()}
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            return await response.json()

    async def fetch_image_bytes(self, session: aiohttp.ClientSession, url: str, idx: int) -> bytes:
        print(f"Downloading image {idx} started")
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.read()
        print(f"Downloading image {idx} finished")
        return data

    async def save_file(self, path: str, data: bytes):
        async with aiofiles.open(path, "wb") as f:
            await f.write(data)
        # Для numpy массивов (cv2) aiofiles не подходит напрямую, нужно энкодить.
        # Но мы сохраняем исходные байты через aiofiles.

    async def save_cv2_image(self, path: str, image):
        # cv2.imencode возвращает (success, buffer)
        success, buffer = cv2.imencode(os.path.splitext(path)[1], image)
        if success:
            async with aiofiles.open(path, "wb") as f:
                await f.write(buffer.tobytes())

    async def process_pipeline(self):
        start_total = time.perf_counter()
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Получаем список URL (один запрос, можно не делать генератором)
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_json(session)
        
        # Подготовка задач
        tasks_data = []
        for i, item in enumerate(data, start=1):
            breeds = item.get("breeds")
            if not breeds: continue
            breed_name = breeds[0].get("name", "unknown")
            url = item.get("url")
            if not url: continue
            
            # Санитизация имени
            s_breed = self._sanitize(breed_name)
            tasks_data.append((i, url, breed_name, s_breed))

        # Используем ProcessPoolExecutor
        loop = asyncio.get_running_loop()
        
        async with aiohttp.ClientSession() as session:
            with ProcessPoolExecutor() as pool:
                # Пайплайн:
                # Скачивание -> Обработка (CPU) -> Сохранение (IO)
                
                # Мы можем запустить скачивания
                download_tasks = []
                for idx, url, breed, s_breed in tasks_data:
                    download_tasks.append(self._process_single_image(session, pool, idx, url, breed, s_breed))
                
                await asyncio.gather(*download_tasks)

        end_total = time.perf_counter()
        print(f"Total execution time: {end_total - start_total:.3f}s")

    async def _process_single_image(self, session, pool, idx, url, breed, s_breed):
        # 1. Скачивание
        img_bytes = await self.fetch_image_bytes(session, url, idx)
        
        # 2. Обработка в процессе
        # run_in_executor запускает функцию в пуле
        loop = asyncio.get_running_loop()
        cat_img, edges_lib, edges_custom = await loop.run_in_executor(
            pool, _cpu_bound_edge_detect, img_bytes, url, breed
        )
        
        # 3. Сохранение (асинхронно)
        base = os.path.join(self.output_dir, f"{idx}_{s_breed}")
        orig_path = base + "_original.png"
        lib_path = base + "_edges_lib.png"
        cust_path = base + "_edges_custom.png"

        await asyncio.gather(
            self.save_cv2_image(orig_path, cat_img.image),
            self.save_cv2_image(lib_path, edges_lib),
            self.save_cv2_image(cust_path, edges_custom)
        )
        print(f"Saved images for {idx}_{s_breed}")

    async def process_pipeline_generator(self):
        """
        Реализация через асинхронные генераторы (доп. задание)
        """
        start_total = time.perf_counter()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # ProcessPoolExecutor должен жить все время
        self.pool = ProcessPoolExecutor()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Этап 1: Генератор URL
                url_gen = self._gen_urls(session)
                
                # Этап 2: Генератор скачанных изображений
                download_gen = self._gen_downloads(session, url_gen)
                
                # Этап 3: Генератор обработки (CPU)
                processed_gen = self._gen_processing(download_gen)
                
                # Этап 4: Сохранение (потребление)
                await self._consume_save(processed_gen)
                
        finally:
            self.pool.shutdown()
            
        end_total = time.perf_counter()
        print(f"Total execution time (Generator Pipeline): {end_total - start_total:.3f}s")

    async def _gen_urls(self, session) -> AsyncGenerator[Dict, None]:
        data = await self.fetch_json(session)
        for i, item in enumerate(data, start=1):
            yield {"idx": i, "item": item}

    async def _gen_downloads(self, session, input_gen) -> AsyncGenerator[Dict, None]:
        # Для параллельного скачивания нужно запускать задачи, а не ждать по одной
        # Но в "чистом" пайплайне генераторов часто данные идут последовательно или батчами.
        # Чтобы сделать true async pipeline, можно использовать asyncio.Queue или map.
        # Здесь реализуем простую схему: yield по мере готовности.
        
        # Чтобы скачивать параллельно внутри генератора, можно набрать батч или запустить таски.
        # Упрощенный вариант:
        tasks = []
        async for data in input_gen:
            idx = data["idx"]
            item = data["item"]
            breeds = item.get("breeds")
            if not breeds: continue
            breed = breeds[0].get("name", "unknown")
            url = item.get("url")
            if not url: continue
            
            # Запускаем скачивание сразу
            task = asyncio.create_task(self._fetch_wrapper(session, url, idx, breed))
            tasks.append(task)
            
        for task in asyncio.as_completed(tasks):
            yield await task

    async def _fetch_wrapper(self, session, url, idx, breed):
        data = await self.fetch_image_bytes(session, url, idx)
        return {"idx": idx, "data": data, "url": url, "breed": breed}

    async def _gen_processing(self, input_gen) -> AsyncGenerator[Dict, None]:
        tasks = []
        loop = asyncio.get_running_loop()
        
        async for item in input_gen:
            # Запускаем обработку в пуле
            future = loop.run_in_executor(
                self.pool, 
                _cpu_bound_edge_detect, 
                item["data"], item["url"], item["breed"]
            )
            tasks.append((future, item["idx"], item["breed"]))
            
        # Ждем результаты. Порядок может нарушиться, но idx у нас сохранен.
        # as_completed для futures сложнее, проще собрать awaitable обертки
        
        # Переделаем: создадим список asyncio.Task, которые ждут executor future
        async def wait_proc(fut, idx, breed):
            res = await fut
            return {"idx": idx, "breed": breed, "result": res}
            
        proc_tasks = [asyncio.create_task(wait_proc(f, i, b)) for f, i, b in tasks]
        
        for t in asyncio.as_completed(proc_tasks):
            yield await t

    async def _consume_save(self, input_gen):
        save_tasks = []
        async for item in input_gen:
            idx = item["idx"]
            breed = item["breed"]
            cat_img, lib, cust = item["result"]
            s_breed = self._sanitize(breed)
            
            base = os.path.join(self.output_dir, f"{idx}_{s_breed}")
            
            save_tasks.append(self.save_cv2_image(base + "_original.png", cat_img.image))
            save_tasks.append(self.save_cv2_image(base + "_edges_lib.png", lib))
            save_tasks.append(self.save_cv2_image(base + "_edges_custom.png", cust))
            
        await asyncio.gather(*save_tasks)
