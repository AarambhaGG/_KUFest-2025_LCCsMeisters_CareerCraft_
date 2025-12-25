"""
Views for Jobs app
"""
import json
from django.http import StreamingHttpResponse
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django_filters.rest_framework import DjangoFilterBackend
from drf_spectacular.utils import extend_schema, extend_schema_view

from .models import Job, JobEligibilityAnalysis
from .serializers import (
    JobListSerializer,
    JobDetailSerializer,
    JobEligibilityAnalysisSerializer,
    JobEligibilityAnalysisDetailSerializer,
    AnalyzeJobEligibilitySerializer,
    ReanalyzeJobEligibilitySerializer,
)
from .services import JobEligibilityAnalyzer, DreamJobParser, AnalysisChatService
from .streaming_services import StreamingJobAnalyzer


@extend_schema_view(
    list=extend_schema(
        tags=['Jobs'],
        summary='List all jobs',
        description='Get a paginated list of all active jobs with filtering',
    ),
    retrieve=extend_schema(
        tags=['Jobs'],
        summary='Get job details',
        description='Get detailed information about a specific job',
    ),
)
class JobViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for browsing jobs
    """
    queryset = Job.objects.filter(status='ACTIVE').select_related(
        'added_by'
    ).prefetch_related('skill_requirements__skill')
    permission_classes = [AllowAny]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ['job_type', 'experience_level', 'is_remote', 'remote_policy']
    search_fields = ['title', 'company_name', 'description', 'requirements']
    ordering_fields = ['created_at', 'posted_date', 'title', 'salary_min']
    ordering = ['-created_at']

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return JobDetailSerializer
        return JobListSerializer

    def retrieve(self, request, *args, **kwargs):
        """Increment view count when job is viewed"""
        instance = self.get_object()
        instance.view_count += 1
        instance.save(update_fields=['view_count'])
        serializer = self.get_serializer(instance)
        return Response(serializer.data)

    @extend_schema(
        tags=['Jobs'],
        summary='List analyzed jobs',
        description='Get all jobs that the current user has analyzed, with their latest analysis data',
    )
    @action(detail=False, methods=['get'], permission_classes=[IsAuthenticated])
    def analyzed(self, request):
        """
        Get all jobs that the user has analyzed, with the latest analysis for each
        """
        from django.db.models import Max

        # Get the latest analysis for each job
        latest_analyses = (
            JobEligibilityAnalysis.objects
            .filter(user=request.user)
            .values('job')
            .annotate(latest_id=Max('id'))
            .values_list('latest_id', flat=True)
        )

        # Get those analyses with job data
        analyses = (
            JobEligibilityAnalysis.objects
            .filter(id__in=latest_analyses)
            .select_related('job', 'user')
            .order_by('-analyzed_at')
        )

        # Filter by eligibility level if provided
        eligibility_level = request.query_params.get('eligibility_level')
        if eligibility_level:
            analyses = analyses.filter(eligibility_level=eligibility_level)

        page = self.paginate_queryset(analyses)
        if page is not None:
            data = []
            for analysis in page:
                job_data = JobListSerializer(analysis.job).data
                job_data['latest_analysis'] = JobEligibilityAnalysisSerializer(analysis).data
                data.append(job_data)
            return self.get_paginated_response(data)

        data = []
        for analysis in analyses:
            job_data = JobListSerializer(analysis.job).data
            job_data['latest_analysis'] = JobEligibilityAnalysisSerializer(analysis).data
            data.append(job_data)

        return Response(data)

    @extend_schema(
        tags=['Jobs'],
        summary='Analyze job eligibility',
        description='Analyze how well the current user matches this job using AI',
        request=AnalyzeJobEligibilitySerializer,
        responses={200: JobEligibilityAnalysisDetailSerializer},
    )
    @action(detail=True, methods=['post'], permission_classes=[IsAuthenticated])
    def analyze_eligibility(self, request, pk=None):
        """
        Analyze user's eligibility for this job using LangChain AI
        """
        job = self.get_object()
        serializer = AnalyzeJobEligibilitySerializer(data={'job_id': job.id, **request.data})
        serializer.is_valid(raise_exception=True)

        additional_context = serializer.validated_data.get('additional_context', '')

        try:
            # Initialize analyzer
            analyzer = JobEligibilityAnalyzer(model_name="gpt-4")

            # Perform analysis
            analysis = analyzer.analyze_eligibility(
                user=request.user,
                job=job,
                additional_context=additional_context
            )

            # Return analysis
            response_serializer = JobEligibilityAnalysisDetailSerializer(analysis)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Exception as e:
            return Response(
                {'error': f'Analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


@extend_schema_view(
    list=extend_schema(
        tags=['Job Analysis'],
        summary='List my job analyses',
        description='Get all job eligibility analyses for the current user',
    ),
    retrieve=extend_schema(
        tags=['Job Analysis'],
        summary='Get analysis details',
        description='Get detailed information about a specific job analysis',
    ),
)
class JobEligibilityAnalysisViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing job eligibility analyses
    """
    permission_classes = [IsAuthenticated]
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['eligibility_level', 'job']
    ordering_fields = ['analyzed_at', 'match_score']
    ordering = ['-analyzed_at']

    def get_queryset(self):
        """Filter to current user's analyses"""
        return JobEligibilityAnalysis.objects.filter(
            user=self.request.user
        ).select_related('job', 'user')

    def list(self, request, *args, **kwargs):
        """
        List analyses grouped by job posting.
        Returns only the most recent analysis for each unique job.
        """
        from django.db.models import Max

        # Get all user's analyses
        queryset = self.filter_queryset(self.get_queryset())

        # Get the latest analysis ID for each job
        latest_analyses = (
            queryset
            .values('job')
            .annotate(latest_id=Max('id'))
            .values_list('latest_id', flat=True)
        )

        # Filter to only the latest analyses
        queryset = queryset.filter(id__in=latest_analyses).order_by('-analyzed_at')

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def get_serializer_class(self):
        if self.action == 'retrieve':
            return JobEligibilityAnalysisDetailSerializer
        return JobEligibilityAnalysisSerializer

    @extend_schema(
        tags=['Job Analysis'],
        summary='Get all analyses for a specific job',
        description='Get analysis history for a specific job (all analyses, not just the latest)',
    )
    @action(detail=False, methods=['get'], url_path='by-job/(?P<job_id>[^/.]+)')
    def by_job(self, request, job_id=None):
        """
        Get all analyses for a specific job, ordered by date (newest first)
        """
        queryset = self.get_queryset().filter(job_id=job_id).order_by('-analyzed_at')

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    @extend_schema(
        tags=['Job Analysis'],
        summary='Analyze job eligibility',
        description='Analyze eligibility for a job with optional additional context',
        request=AnalyzeJobEligibilitySerializer,
        responses={200: JobEligibilityAnalysisDetailSerializer},
    )
    @action(detail=False, methods=['post'])
    def analyze(self, request):
        """
        Analyze user's eligibility for a job
        """
        serializer = AnalyzeJobEligibilitySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        job_id = serializer.validated_data['job_id']
        additional_context = serializer.validated_data.get('additional_context', '')

        try:
            job = Job.objects.get(id=job_id)

            # Initialize analyzer
            analyzer = JobEligibilityAnalyzer(model_name="gpt-4")

            # Perform analysis
            analysis = analyzer.analyze_eligibility(
                user=request.user,
                job=job,
                additional_context=additional_context
            )

            # Return analysis
            response_serializer = JobEligibilityAnalysisDetailSerializer(analysis)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except Job.DoesNotExist:
            return Response(
                {'error': 'Job not found'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'Analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        tags=['Job Analysis'],
        summary='Re-analyze with additional context',
        description='Re-run analysis with additional context from user',
        request=ReanalyzeJobEligibilitySerializer,
        responses={200: JobEligibilityAnalysisDetailSerializer},
    )
    @action(detail=False, methods=['post'])
    def reanalyze(self, request):
        """
        Re-analyze with additional context
        """
        serializer = ReanalyzeJobEligibilitySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        analysis_id = serializer.validated_data['analysis_id']
        additional_context = serializer.validated_data['additional_context']

        try:
            # Get previous analysis
            previous_analysis = JobEligibilityAnalysis.objects.get(
                id=analysis_id,
                user=request.user
            )

            # Initialize analyzer
            analyzer = JobEligibilityAnalyzer(model_name="gpt-4")

            # Re-analyze
            new_analysis = analyzer.reanalyze_with_context(
                analysis=previous_analysis,
                additional_context=additional_context
            )

            # Return analysis
            response_serializer = JobEligibilityAnalysisDetailSerializer(new_analysis)
            return Response(response_serializer.data, status=status.HTTP_200_OK)

        except JobEligibilityAnalysis.DoesNotExist:
            return Response(
                {'error': 'Analysis not found or you do not have permission to access it'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'Re-analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        tags=['Job Analysis'],
        summary='Get analysis statistics',
        description='Get statistics about user\'s job analyses',
    )
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """
        Get statistics about user's job analyses
        """
        analyses = self.get_queryset()

        stats = {
            'total_analyses': analyses.count(),
            'by_eligibility_level': {
                'EXCELLENT': analyses.filter(eligibility_level='EXCELLENT').count(),
                'GOOD': analyses.filter(eligibility_level='GOOD').count(),
                'FAIR': analyses.filter(eligibility_level='FAIR').count(),
                'POOR': analyses.filter(eligibility_level='POOR').count(),
            },
            'average_match_score': 0,
            'recent_analyses': JobEligibilityAnalysisSerializer(
                analyses[:5], many=True
            ).data,
        }

        # Calculate average match score
        if analyses.exists():
            from django.db.models import Avg
            avg_score = analyses.aggregate(Avg('match_score'))['match_score__avg']
            stats['average_match_score'] = round(avg_score, 2) if avg_score else 0

        return Response(stats)

    @extend_schema(
        tags=['Job Analysis'],
        summary='Chat about job analysis',
        description='Ask questions about a specific job analysis and get AI-powered answers',
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'analysis_id': {
                        'type': 'integer',
                        'description': 'ID of the job analysis to chat about',
                    },
                    'message': {
                        'type': 'string',
                        'description': 'User question about the analysis',
                    },
                },
                'required': ['analysis_id', 'message'],
            }
        },
    )
    @action(detail=False, methods=['post'])
    def chat(self, request):
        """
        Chat with AI about a specific job analysis

        Ask questions like:
        - "What should I focus on first?"
        - "How long will it take to be job-ready?"
        - "What are the most important skills to learn?"
        - "Can you explain my skill gaps in detail?"
        """
        analysis_id = request.data.get('analysis_id')
        message = request.data.get('message')

        if not analysis_id or not message:
            return Response(
                {'error': 'analysis_id and message are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get analysis
            analysis = JobEligibilityAnalysis.objects.get(
                id=analysis_id,
                user=request.user
            )

            # Initialize chat service
            chat_service = AnalysisChatService()

            # Get response from AI
            response = chat_service.chat_about_analysis(
                analysis=analysis,
                message=message
            )

            return Response({
                'message': message,
                'response': response,
            }, status=status.HTTP_200_OK)

        except JobEligibilityAnalysis.DoesNotExist:
            return Response(
                {'error': 'Analysis not found or you do not have permission to access it'},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {'error': f'Chat failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        tags=['Job Analysis'],
        summary='Analyze dream job',
        description='Analyze eligibility for a dream job described by the user or pasted job description',
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'job_description': {
                        'type': 'string',
                        'description': 'Job description text (from job board) or dream job description',
                    },
                    'additional_context': {
                        'type': 'string',
                        'description': 'Optional additional context about user skills/experience',
                    },
                    'save_job': {
                        'type': 'boolean',
                        'description': 'Whether to save the parsed job to database (default: false)',
                        'default': False,
                    },
                },
                'required': ['job_description'],
            }
        },
        responses={200: JobEligibilityAnalysisDetailSerializer},
    )
    @action(detail=False, methods=['post'])
    def analyze_dream_job(self, request):
        """
        Analyze user's eligibility for their dream job

        User can either:
        1. Paste a job description from a job board
        2. Describe their dream job in natural language

        AI will:
        1. Parse the job description to extract requirements
        2. Create a temporary job object
        3. Analyze user's eligibility
        4. Optionally save the job to database
        """
        job_description = request.data.get('job_description')
        additional_context = request.data.get('additional_context', '')
        save_job = request.data.get('save_job', False)

        if not job_description:
            return Response(
                {'error': 'job_description is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Step 1: Parse job description with AI
            parser = DreamJobParser()
            parsed_job_data = parser.parse_job_description(job_description)

            # Step 2: Create job object (saved or temporary)
            if save_job:
                # Map remote policy to correct values
                remote_policy = parsed_job_data.get("remote_policy", "REMOTE")
                if remote_policy == "FULLY_REMOTE":
                    remote_policy = "REMOTE"
                elif remote_policy == "ON_SITE":
                    remote_policy = "ONSITE"

                # Generate unique source URL for saved dream jobs
                import uuid
                source_url = f"https://skillsetz.com/dream-jobs/{uuid.uuid4()}"

                # Build job kwargs, only include salary_currency if provided
                job_kwargs = {
                    "title": parsed_job_data.get("job_title", "Dream Job"),
                    "company_name": parsed_job_data.get("company_name", "Dream Company"),
                    "company_description": parsed_job_data.get("company_culture", ""),
                    "job_type": parsed_job_data.get("job_type", "FULL_TIME"),
                    "experience_level": parsed_job_data.get("experience_level", "MID"),
                    "location": parsed_job_data.get("location", "Remote"),
                    "is_remote": parsed_job_data.get("is_remote", True),
                    "remote_policy": remote_policy,
                    "description": parsed_job_data.get("description", ""),
                    "responsibilities": "\n".join(
                        [f"• {resp}" for resp in parsed_job_data.get("responsibilities", [])]
                    ) if isinstance(parsed_job_data.get("responsibilities"), list) else str(parsed_job_data.get("responsibilities", "")),
                    "requirements": "\n".join(
                        [f"• {req.get('name', req) if isinstance(req, dict) else req}"
                         for req in parsed_job_data.get("required_skills", [])]
                    ),
                    "salary_min": parsed_job_data.get("min_salary"),
                    "salary_max": parsed_job_data.get("max_salary"),
                    "source_url": source_url,
                    "source_platform": "Dream Job (User Created)",
                    "parsed_skills": parsed_job_data.get("required_skills", [])
                    + parsed_job_data.get("preferred_skills", []),
                    "parsed_requirements": parsed_job_data,
                    "status": 'ACTIVE',
                    "added_by": request.user,
                }

                # Only add salary_currency if provided (let model default handle it otherwise)
                if parsed_job_data.get("salary_currency"):
                    job_kwargs["salary_currency"] = parsed_job_data.get("salary_currency")

                # Save to database
                job = Job.objects.create(**job_kwargs)
            else:
                # Create temporary job (not saved)
                job = parser.create_temporary_job(parsed_job_data)

            # Step 3: Analyze eligibility
            analyzer = JobEligibilityAnalyzer(model_name="gpt-4")
            analysis = analyzer.analyze_eligibility(
                user=request.user,
                job=job,
                additional_context=additional_context
            )

            # Step 4: Return results
            response_data = {
                'message': 'Dream job analyzed successfully',
                'parsed_job': parsed_job_data,
                'job_saved': save_job,
                'analysis': JobEligibilityAnalysisDetailSerializer(analysis).data,
            }

            if save_job:
                response_data['job_id'] = job.id
                response_data['job_url'] = f'/api/jobs/{job.id}/'

            return Response(response_data, status=status.HTTP_200_OK)

        except ValueError as e:
            return Response(
                {'error': str(e)},
                status=status.HTTP_400_BAD_REQUEST
            )
        except Exception as e:
            return Response(
                {'error': f'Dream job analysis failed: {str(e)}'},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

    @extend_schema(
        tags=['Job Analysis'],
        summary='Stream analyze dream job (SSE)',
        description='Stream job analysis with real-time updates using Server-Sent Events',
        request={
            'application/json': {
                'type': 'object',
                'properties': {
                    'job_description': {
                        'type': 'string',
                        'description': 'Job description text or dream job description',
                    },
                    'additional_context': {
                        'type': 'string',
                        'description': 'Optional additional context',
                    },
                    'save_job': {
                        'type': 'boolean',
                        'description': 'Whether to save the job (default: false)',
                        'default': False,
                    },
                },
                'required': ['job_description'],
            }
        },
    )
    @action(detail=False, methods=['post'])
    def stream_analyze_dream_job(self, request):
        """
        Stream job analysis with real-time updates using Server-Sent Events

        Returns a stream of events:
        - status: Progress updates
        - partial_metric: Metrics as they're calculated
        - metrics_complete: All metrics calculated
        - complete: Analysis finished
        - error: Error occurred
        """
        job_description = request.data.get('job_description')
        additional_context = request.data.get('additional_context', '')
        save_job = request.data.get('save_job', False)

        if not job_description:
            return Response(
                {'error': 'job_description is required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        def event_stream():
            """Generator for SSE events"""
            try:
                # Step 1: Parse job description
                yield f"data: {json.dumps({'type': 'status', 'step': 'parsing', 'message': 'Parsing job description with AI...', 'progress': 5})}\n\n"

                parser = DreamJobParser()
                parsed_job_data = parser.parse_job_description(job_description)

                yield f"data: {json.dumps({'type': 'status', 'step': 'parsed', 'message': 'Job description parsed successfully', 'progress': 15})}\n\n"

                # Step 2: Create job object
                if save_job:
                    remote_policy = parsed_job_data.get("remote_policy", "REMOTE")
                    if remote_policy == "FULLY_REMOTE":
                        remote_policy = "REMOTE"
                    elif remote_policy == "ON_SITE":
                        remote_policy = "ONSITE"

                    import uuid
                    source_url = f"https://skillsetz.com/dream-jobs/{uuid.uuid4()}"

                    job_kwargs = {
                        "title": parsed_job_data.get("job_title", "Dream Job"),
                        "company_name": parsed_job_data.get("company_name", "Dream Company"),
                        "company_description": parsed_job_data.get("company_culture", ""),
                        "job_type": parsed_job_data.get("job_type", "FULL_TIME"),
                        "experience_level": parsed_job_data.get("experience_level", "MID"),
                        "location": parsed_job_data.get("location", "Remote"),
                        "is_remote": parsed_job_data.get("is_remote", True),
                        "remote_policy": remote_policy,
                        "description": parsed_job_data.get("description", ""),
                        "responsibilities": "\n".join(
                            [f"• {resp}" for resp in parsed_job_data.get("responsibilities", [])]
                        ) if isinstance(parsed_job_data.get("responsibilities"), list) else str(parsed_job_data.get("responsibilities", "")),
                        "requirements": "\n".join(
                            [f"• {req.get('name', req) if isinstance(req, dict) else req}"
                             for req in parsed_job_data.get("required_skills", [])]
                        ),
                        "salary_min": parsed_job_data.get("min_salary"),
                        "salary_max": parsed_job_data.get("max_salary"),
                        "source_url": source_url,
                        "source_platform": "Dream Job (User Created)",
                        "parsed_skills": parsed_job_data.get("required_skills", []) + parsed_job_data.get("preferred_skills", []),
                        "parsed_requirements": parsed_job_data,
                        "status": 'ACTIVE',
                        "added_by": request.user,
                    }

                    if parsed_job_data.get("salary_currency"):
                        job_kwargs["salary_currency"] = parsed_job_data.get("salary_currency")

                    job = Job.objects.create(**job_kwargs)
                    job_id = job.id
                else:
                    job = parser.create_temporary_job(parsed_job_data)
                    job_id = None

                # Step 3: Stream analysis
                analyzer = StreamingJobAnalyzer()

                for event in analyzer.stream_analyze_eligibility(
                    user=request.user,
                    job=job,
                    additional_context=additional_context
                ):
                    # Forward all events to client
                    yield f"data: {json.dumps(event)}\n\n"

                # Step 4: Send final data with job info
                final_event = {
                    'type': 'final',
                    'parsed_job': parsed_job_data,
                    'job_saved': save_job,
                    'job_id': job_id,
                    'job_url': f'/api/jobs/{job_id}/' if job_id else None,
                }
                yield f"data: {json.dumps(final_event)}\n\n"

            except Exception as e:
                error_event = {
                    'type': 'error',
                    'error': str(e),
                    'message': f'Stream analysis failed: {str(e)}'
                }
                yield f"data: {json.dumps(error_event)}\n\n"

        response = StreamingHttpResponse(
            event_stream(),
            content_type='text/event-stream'
        )
        response['Cache-Control'] = 'no-cache'
        response['X-Accel-Buffering'] = 'no'
        return response
