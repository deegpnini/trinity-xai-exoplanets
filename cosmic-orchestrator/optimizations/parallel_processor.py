import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional
import logging
import time
import numpy as np

class ParallelProcessor:
    """
    Sistema avançado de processamento paralelo para chamadas multi-IA
    Otimiza dramaticamente o pipeline do Cosmic Orchestrator
    """
    
    def __init__(self, max_concurrent_ias: int = 3, timeout: float = 30.0):
        self.max_concurrent_ias = max_concurrent_ias
        self.timeout = timeout
        self.session: Optional[aiohttp.ClientSession] = None
        self.thread_pool = ThreadPoolExecutor(max_workers=max_concurrent_ias)
        self.logger = logging.getLogger('CosmicOrchestrator')
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0
        }
    
    async def __aenter__(self):
        """Context manager para gerenciamento de sessão"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup da sessão"""
        if self.session:
            await self.session.close()
    
    async def process_ia_calls_parallel(self, 
                                      trinity_result: Dict[str, Any],
                                      ia_services: List) -> Dict[str, Any]:
        """
        Processa múltiplas chamadas de IA em paralelo com timeout
        """
        self.metrics['total_calls'] += len(ia_services)
        start_time = time.time()
        
        try:
            # Criar tasks para todas as IAs
            tasks = []
            for ia_service in ia_services:
                task = self._create_ia_task(ia_service, trinity_result)
                tasks.append(task)
            
            # Executar com timeout
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.timeout
            )
            
            # Processar resultados
            processed_results = self._process_ia_results(results, ia_services)
            self._update_metrics(start_time, len(ia_services))
            
            return processed_results
            
        except asyncio.TimeoutError:
            self.logger.warning(f"IA parallel processing timeout after {self.timeout}s")
            return self._handle_timeout(ia_services)
        except Exception as e:
            self.logger.error(f"Parallel processing failed: {e}")
            return self._handle_critical_error(e)
    
    async def _create_ia_task(self, ia_service, data: Dict) -> Dict[str, Any]:
        """
        Cria task assíncrona para chamada de IA
        """
        try:
            # Verificar se a IA suporta async
            if hasattr(ia_service, 'analyze_async') and callable(ia_service.analyze_async):
                return await ia_service.analyze_async(data)
            else:
                # Executar versão síncrona em thread separada
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.thread_pool, 
                    ia_service.analyze, 
                    data
                )
        except Exception as e:
            self.logger.error(f"IA task creation failed for {getattr(ia_service, 'name', 'unknown')}: {e}")
            return {"error": str(e), "source": getattr(ia_service, 'name', 'unknown')}
    
    def _process_ia_results(self, results: List, ia_services: List) -> Dict[str, Any]:
        """
        Processa e combina resultados das IAs
        """
        successful_results = []
        failed_services = []
        
        for i, result in enumerate(results):
            service_name = getattr(ia_services[i], 'name', f'ia_{i}')
            
            if isinstance(result, Exception):
                self.logger.warning(f"IA {service_name} raised exception: {result}")
                failed_services.append(service_name)
            elif result.get('error'):
                self.logger.warning(f"IA {service_name} returned error: {result.get('error')}")
                failed_services.append(service_name)
            else:
                successful_results.append(result)
                self.metrics['successful_calls'] += 1
        
        self.metrics['failed_calls'] += len(failed_services)
        
        # Calcular consenso entre resultados válidos
        consensus_analysis = self._analyze_consensus(successful_results)
        
        return {
            "successful_ias": len(successful_results),
            "failed_ias": len(failed_services),
            "failed_services": failed_services,
            "consensus_analysis": consensus_analysis,
            "individual_results": successful_results,
            "final_decision": self._make_final_decision(successful_results),
            "confidence_score": consensus_analysis.get('confidence', 0.0)
        }
    
    def _analyze_consensus(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Analisa consenso entre resultados de IA
        """
        if not results:
            return {"confidence": 0.0, "agreement_level": "none", "disagreement_score": 1.0}
        
        # Extrair scores de bioassinatura
        biosignature_scores = []
        confidence_scores = []
        
        for result in results:
            bs_score = result.get('biosignature_score')
            conf_score = result.get('confidence', 0.5)
            
            if bs_score is not None:
                biosignature_scores.append(bs_score)
                confidence_scores.append(conf_score)
        
        if not biosignature_scores:
            return {"confidence": 0.0, "agreement_level": "inconclusive"}
        
        # Calcular métricas de consenso
        avg_score = np.mean(biosignature_scores)
        weighted_avg = np.average(biosignature_scores, weights=confidence_scores)
        
        # Calcular variância (medida de desacordo)
        variance = np.var(biosignature_scores)
        max_variance = 0.25  # Máxima variância possível para scores 0-1
        disagreement_score = min(1.0, variance / max_variance)
        
        # Nível de concordância baseado na variância
        if variance < 0.01:
            agreement_level = "high"
            confidence_boost = 0.2
        elif variance < 0.05:
            agreement_level = "medium"
            confidence_boost = 0.1
        else:
            agreement_level = "low"
            confidence_boost = -0.1
        
        base_confidence = np.mean(confidence_scores)
        final_confidence = max(0.0, min(1.0, base_confidence + confidence_boost))
        
        return {
            "average_score": float(avg_score),
            "weighted_score": float(weighted_avg),
            "confidence": float(final_confidence),
            "agreement_level": agreement_level,
            "disagreement_score": float(disagreement_score),
            "score_std_dev": float(np.std(biosignature_scores))
        }
    
    def _make_final_decision(self, results: List[Dict]) -> Dict[str, Any]:
        """
        Toma decisão final baseada em consenso
        """
        if not results:
            return {
                "biosignature_score": 0.0,
                "confidence": 0.0,
                "classification": "inconclusive",
                "reason": "no_successful_ia_results"
            }
        
        # Usar análise de consenso para decisão final
        consensus = self._analyze_consensus(results)
        final_score = consensus['weighted_score']
        confidence = consensus['confidence']
        
        # Classificação baseada no score e confiança
        if final_score > 0.8 and confidence > 0.7:
            classification = "high_potential"
        elif final_score > 0.5 and confidence > 0.5:
            classification = "medium_potential"
        elif final_score > 0.3:
            classification = "low_potential"
        else:
            classification = "unlikely"
        
        return {
            "biosignature_score": final_score,
            "confidence": confidence,
            "classification": classification,
            "consensus_quality": consensus['agreement_level']
        }
    
    def _handle_timeout(self, ia_services: List) -> Dict[str, Any]:
        """Lida com timeout nas chamadas de IA"""
        service_names = [getattr(service, 'name', 'unknown') for service in ia_services]
        
        return {
            "error": "parallel_processing_timeout",
            "failed_services": service_names,
            "successful_ias": 0,
            "failed_ias": len(ia_services),
            "final_decision": {
                "biosignature_score": 0.0,
                "confidence": 0.0,
                "classification": "timeout_error",
                "reason": "ia_calls_timeout"
            }
        }
    
    def _handle_critical_error(self, error: Exception) -> Dict[str, Any]:
        """Lida com erros críticos no processamento paralelo"""
        return {
            "error": f"critical_parallel_error: {str(error)}",
            "successful_ias": 0,
            "failed_ias": "all",
            "final_decision": {
                "biosignature_score": 0.0,
                "confidence": 0.0,
                "classification": "processing_error",
                "reason": "parallel_system_failure"
            }
        }
    
    def _update_metrics(self, start_time: float, total_calls: int):
        """Atualiza métricas de performance"""
        processing_time = time.time() - start_time
        avg_time = processing_time / total_calls if total_calls > 0 else 0
        
        # Média móvel para response time
        if self.metrics['average_response_time'] == 0:
            self.metrics['average_response_time'] = avg_time
        else:
            self.metrics['average_response_time'] = (
                0.7 * self.metrics['average_response_time'] + 0.3 * avg_time
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais do processador"""
        success_rate = (
            self.metrics['successful_calls'] / self.metrics['total_calls'] 
            if self.metrics['total_calls'] > 0 else 0
        )
        
        return {
            **self.metrics,
            "success_rate": success_rate,
            "failure_rate": 1 - success_rate,
            "concurrent_capacity": self.max_concurrent_ias
        }
    
    def reset_metrics(self):
        """Reseta métricas para novo ciclo"""
        self.metrics = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'average_response_time': 0.0
        }