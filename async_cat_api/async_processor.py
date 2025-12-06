import os
import time
import asyncio
from typing import List, Dict, AsyncGenerator, Tuple
from concurrent.futures import ProcessPoolExecutor

import aiohttp
import aiofiles
import cv2

from cat_api.cat_image import CatImage
from cat_api.processor import CatImageProcessor


def _cpu_bound_edge_detect(img_bytes: bytes, url: str, breed: str) -> Tuple["CatImage", object, object]:
    """
    CPU-bound функция для выполнения в отдельном процессе.
    Выполняет десериализацию изображения и свертку (выделение границ).
    """
    pid = os.getpid()
    print(f"Convolution for image '{breed}' started (PID {pid})")
    
    cat_img = CatImage.from_bytes(img_bytes, url, breed)
    
    # Предполагается, что методы возвращают numpy массивы или объекты, совместимые с cv2
    edges_lib = cat_img.edge_detect_library()
    edges_custom = cat_img.edge_detect_custom()
    
    print(f"Convolution for image '{breed}' finished (PID {pid})")
    return cat_img, edges_lib, edges_custom


class AsyncCatImageProcessor(CatImageProcessor):
    """
    Асинхронный процессор для скачивания и обработки изображений котиков.
    Поддерживает параллельную обработку через ProcessPoolExecutor.
    """

    def __init__(self, provider: str = "cat", output_dir: str = "downloads", limit: int = 1):
        super().__init__(provider, output_dir, limit)
        self.pool = None  # Будет инициализирован при необходимости

    async def fetch_json(self, session: aiohttp.ClientSession) -> List[Dict]:
        """Асинхронное получение списка URL через API."""
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
        """Асинхронное скачивание байтов изображения."""
        print(f"Downloading image {idx} started")
        async with session.get(url) as response:
            response.raise_for_status()
            data = await response.read()
        print(f"Downloading image {idx} finished")
        return data

    async def save_cv2_image(self, path: str, image):
        """
        Асинхронное сохранение изображения OpenCV.
        Кодирование (imencode) выполняется синхронно (быстрая операция),
        запись на диск - асинхронно.
        """
        # Определяем расширение и кодируем в буфер памяти
        ext = os.path.splitext(path)[1]
        success, buffer = cv2.imencode(ext, image)
        
        if success:
            async with aiofiles.open(path, "wb") as f:
                await f.write(buffer.tobytes())

    async def process_pipeline(self):
        """
        Стандартный режим: Gather-based подход.
        1. Получаем список URL.
        2. Формируем задачи на скачивание + обработку + сохранение.
        3. Запускаем всё через asyncio.gather.
        """
        print("Starting Standard Async Pipeline...")
        start_total = time.perf_counter()
        os.makedirs(self.output_dir, exist_ok=True)

        # 1. Получение метаданных
        async with aiohttp.ClientSession() as session:
            data = await self.fetch_json(session)
        
        # Подготовка данных задач
        tasks_data = []
        for i, item in enumerate(data, start=1):
            breeds = item.get("breeds")
            if not breeds: continue
            breed_name = breeds[0].get("name", "unknown")
            url = item.get("url")
            if not url: continue
            
            s_breed = self._sanitize(breed_name)
            tasks_data.append((i, url, breed_name, s_breed))

        # Запуск обработки
        async with aiohttp.ClientSession() as session:
            with ProcessPoolExecutor() as pool:
                tasks = []
                for idx, url, breed, s_breed in tasks_data:
                    # Создаем задачу для каждого изображения
                    task = asyncio.create_task(
                        self._process_single_image_standard(session, pool, idx, url, breed, s_breed)
                    )
                    tasks.append(task)
                
                if tasks:
                    await asyncio.gather(*tasks)

        end_total = time.perf_counter()
        print(f"Total execution time (Standard): {end_total - start_total:.3f}s")

    async def _process_single_image_standard(self, session, pool, idx, url, breed, s_breed):
        """Полный цикл обработки одного изображения для стандартного режима."""
        # 1. Скачивание
        img_bytes = await self.fetch_image_bytes(session, url, idx)
        
        # 2. Обработка (CPU)
        loop = asyncio.get_running_loop()
        cat_img, edges_lib, edges_custom = await loop.run_in_executor(
            pool, _cpu_bound_edge_detect, img_bytes, url, breed
        )
        
        # 3. Сохранение (IO)
        base = os.path.join(self.output_dir, f"{idx}_{s_breed}")
        await asyncio.gather(
            self.save_cv2_image(base + "_original.png", cat_img.image),
            self.save_cv2_image(base + "_edges_lib.png", edges_lib),
            self.save_cv2_image(base + "_edges_custom.png", edges_custom)
        )
        print(f"Saved images for {idx}_{s_breed}")

    # -------------------------------------------------------------------------
    # Generator Pipeline Implementation (Дополнительное задание)
    # -------------------------------------------------------------------------

    async def process_pipeline_generator(self):
        """
        Реализация через асинхронные генераторы (конвейер).
        Этапы не блокируют друг друга.
        """
        print("Starting Generator Async Pipeline...")
        start_total = time.perf_counter()
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.pool = ProcessPoolExecutor()
        
        try:
            async with aiohttp.ClientSession() as session:
                # Stage 1: URL Generator
                url_gen = self._gen_urls(session)
                
                # Stage 2: Download Generator (Async IO)
                download_gen = self._gen_downloads(session, url_gen)
                
                # Stage 3: Processing Generator (CPU / Multiprocessing)
                processed_gen = self._gen_processing(download_gen)
                
                # Stage 4: Save Consumer (Async IO)
                await self._consume_save(processed_gen)
                
        finally:
            self.pool.shutdown()
            
        end_total = time.perf_counter()
        print(f"Total execution time (Generator Pipeline): {end_total - start_total:.3f}s")

    async def _gen_urls(self, session) -> AsyncGenerator[Dict, None]:
        """Генератор 1: Выдает URL и метаданные."""
        data = await self.fetch_json(session)
        for i, item in enumerate(data, start=1):
            yield {"idx": i, "item": item}

    async def _gen_downloads(self, session, input_gen) -> AsyncGenerator[Dict, None]:
        """Генератор 2: Скачивает изображения по мере поступления URL."""
        tasks = []
        
        async def _fetch_wrapper(url, idx, breed):
            data = await self.fetch_image_bytes(session, url, idx)
            return {"idx": idx, "data": data, "url": url, "breed": breed}

        async for data in input_gen:
            idx = data["idx"]
            item = data["item"]
            breeds = item.get("breeds")
            if not breeds: continue
            breed = breeds[0].get("name", "unknown")
            url = item.get("url")
            if not url: continue
            
            # Запускаем скачивание немедленно
            task = asyncio.create_task(_fetch_wrapper(url, idx, breed))
            tasks.append(task)
            
        # Возвращаем результаты по мере завершения (as_completed)
        for task in asyncio.as_completed(tasks):
            yield await task

    async def _gen_processing(self, input_gen) -> AsyncGenerator[Dict, None]:
        """Генератор 3: Отправляет скачанные картинки в ProcessPool."""
        tasks = []
        loop = asyncio.get_running_loop()
        
        async def _wait_proc(fut, idx, breed):
            res = await fut
            return {"idx": idx, "breed": breed, "result": res}
        
        async for item in input_gen:
            # Планируем задачу в отдельном процессе
            future = loop.run_in_executor(
                self.pool, 
                _cpu_bound_edge_detect, 
                item["data"], item["url"], item["breed"]
            )
            # Оборачиваем Future в Coroutine Task
            task = asyncio.create_task(_wait_proc(future, item["idx"], item["breed"]))
            tasks.append(task)
        
        for t in asyncio.as_completed(tasks):
            yield await t

    async def _consume_save(self, input_gen):
        """Потребитель: Сохраняет результаты обработки на диск."""
        save_tasks = []
        async for item in input_gen:
            idx = item["idx"]
            breed = item["breed"]
            cat_img, lib, cust = item["result"]
            s_breed = self._sanitize(breed)
            
            base = os.path.join(self.output_dir, f"{idx}_{s_breed}")
            
            # ВАЖНО: Запускаем сохранение сразу через create_task,
            # чтобы не блокировать получение следующего элемента из генератора.
            t1 = asyncio.create_task(self.save_cv2_image(base + "_original.png", cat_img.image))
            t2 = asyncio.create_task(self.save_cv2_image(base + "_edges_lib.png", lib))
            t3 = asyncio.create_task(self.save_cv2_image(base + "_edges_custom.png", cust))
            
            save_tasks.extend([t1, t2, t3])
            
        # Ожидаем завершения всех фоновых задач сохранения
        if save_tasks:
            await asyncio.gather(*save_tasks)