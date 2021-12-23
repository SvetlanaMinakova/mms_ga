class SimJob:
    """
    Simulate job execution
    Attributes:
        task (str): description of task, within which the job is executed
        job (str): description of the job
        processor_name (str): name of the processor, where task is executed
        start_time (float), end_time (float): start and end times of job, executed within task
    """

    def __init__(self, task: str, job: str, processor_name: str, start_time: float, end_time: float):
        self.task = task
        self.job = job
        self.processor_name = processor_name
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return "{task:, " + self.task + ", job: " + self.job + \
               ", processor: " + self.processor_name + \
               ", start: " + str(self.start_time) + ", end: " + str(self.end_time) + "}"


class SimMemoryAccess:
    """
    Simulate access to platform memory
    Attributes:
        task (str): description of task, which accessed the memory
        job (str): description of the job, which accessed the memory within the job
        action (str) : "read" or "write"
        start_time, end_time (float): time, when memory access happened
    """

    def __init__(self, task: str, job: str, mem_name: str, action: str, tokens: int, start_time: float, end_time: float):
        self.task = task
        self.job = job
        self.mem_name = mem_name
        self.action = action
        self.tokens = tokens
        self.start_time = start_time
        self.end_time = end_time

    def __str__(self):
        return "{task:, " + self.task + ", job: " + self.job + \
               ", memory: " + self.mem_name + \
               ", action: " + self.action + \
               ", tokens: " + str(self.tokens) + \
               ", start time: " + str(self.start_time) + ", end_time: " + str(self.end_time) + "}"


class SimMemoryOccupancyInterval:
    """
    Record of time interval during which the platform memory was occupied
    start_step, end_step (int): execution steps in schedule, when memory was occupied

    """
    def __init__(self, start_step: int, end_step: int, max_tokens=0):
        self.start_step = start_step
        self.end_step = end_step
        self.max_tokens = max_tokens

    def __str__(self):
        return "{start step: " + str(self.start_step) + \
               ", end step: " + str(self.end_step) + \
               ", max stored tokens: " + str(self.max_tokens) + \
               "}"


class SimTrace:
    def __init__(self):
        self.jobs = []
        self.memory_accesses = []

    def add_job(self, job: SimJob):
        """
        Add new job
        :param job: simulation job
        """
        self.jobs.append(job)

    def add_mem_access(self, mem_access: SimMemoryAccess):
        """
        Add new memory access
        :param mem_access: memory access
        """
        self.memory_accesses.append(mem_access)

    def get_proc_time(self, proc_name):
        """
            Get time, associated with processor
            If processor is registered within the Trace, end time of
            last job, executed on a processor is returned
            Otherwise, 0 is returned
        :param proc_name (str) unique name of the processor
        """
        proc_time = 0.0
        for job in self.jobs:
            if job.processor_name == proc_name:
                proc_time = max(proc_time, job.end_time)
        return proc_time

    def sort_tasks_by_start_time(self):
        self.jobs.sort(key=lambda task: task.start_time)

    def get_memory_names(self):
        """ Get descriptions of all used memory"""
        memory_names = []
        for mem_access in self.memory_accesses:
            if mem_access.mem_name not in memory_names:
                memory_names.append(mem_access.mem_name)
        return memory_names

    def get_tasks(self):
        """ Get descriptions of all executed tasks"""
        tasks = []
        for job in self.jobs:
            if job.task not in tasks:
                tasks.append(job.task)
        return tasks

    def get_task_times(self):
        """
        Get execution time for every executed task
        :return: dictionary, where key = task (str), value = tuple (start_time, end_time),
        where start_time = time, when task started, end_time = time, when task ended
        """
        task_times = {}
        tasks = self.get_tasks()
        for task in tasks:
            start_time, end_time = self.get_task_time(task)
            task_times[task] = (start_time, end_time)

    def get_task_time(self, task: str):
        """
        Get execution time of a task
        :param task: (string) task
        :return: tuple start_time, end_time where
            - start time is the time when the first job of the task is executed,
            - end time is the time when the last job of the task is executed
        """
        start_time = -1
        end_time = -1
        for job in self.jobs:
            if job.task == task:
                if start_time == -1 or job.start_step < start_time:
                    start_time = job.start_step
                if end_time < job.end_step:
                    end_time = job.end_step
        return start_time, end_time

    def get_task_memories(self, task):
        """ Get names of all memories, ever used by the task"""
        used_memory_names = []
        for memory_access in self.memory_accesses:
            if memory_access.task == task:
                if memory_access.mem_name not in used_memory_names:
                    used_memory_names.append(memory_access.mem_name)
        return used_memory_names

    def get_task_memory_use_time(self, task: str, mem_name: str):
        """
        Get time, during which a task used a memory
        :param task: (string) task
        :param mem_name (string) name of the memory
        :return: tuple start_time, end_time where
            - start time is time, when task first accessed memory
            - end time time, when task last accessed memory
        """
        start_time = -1
        end_time = -1
        for memory_access in self.memory_accesses:
            if memory_access.task == task and memory_access.mem_name == mem_name:
                if start_time == -1 or memory_access.start_time < start_time:
                    start_time = memory_access.start_time
                if memory_access.end_time > end_time:
                    end_time = memory_access.end_time
        return start_time, end_time

    """
    def get_max_stored_tokens(self, mem_name):
        # Get maximum amount of tokens, ever read from memory or written to it
        max_tokens = 0
        for mem_access in self.memory_accesses:
            if mem_access.mem_name == mem_name:
                max_tokens = max(max_tokens, mem_access.tokens)
        return max_tokens
    """

    def get_mem_access_records(self, mem_name):
        """ Get all memory access records, related to memory with specified name
        :param mem_name: memory name
        """
        mem_access_records = []
        for mem_access in self.memory_accesses:
            if mem_access.mem_name == mem_name:
                mem_access_records.append(mem_access)
        return mem_access_records

    def get_max_stored_tokens(self, mem_name):
        """ Get maximum amount of tokens, ever read from memory or written to it"""
        mem_access_records = self.get_mem_access_records(mem_name)
        max_tokens = 0
        stored_tokens = 0
        for record in mem_access_records:
            # upd number of tokens stored
            if record.action == "write":
                stored_tokens += record.tokens
            if record.action == "read":
                stored_tokens -= record.tokens
            max_tokens = max(max_tokens, stored_tokens)
        return max_tokens

    def get_asap_schedule(self):
        """ Get ASAP (CSDF) schedule used for data processing by parts"""
        self.sort_tasks_by_start_time()
        return self.jobs

    def find_step_in_schedule(self, schedule: [SimJob], task, job):
        """
        Find execution step (element id in order) of job in schedule, corresponding to the job
        :param schedule: schedule: set of jobs in execution order
        :param task: task within which the job is performed
        :param job: the job
        :return: (int): execution step (element id in order) of job in schedule, if schedule has
        the job, otherwise -1
        """
        for step in range(len(schedule)):
            if schedule[step].task == task and schedule[step].job == job:
                return step
        return -1

    def generate_memory_occupancy_intervals_per_memory(self, schedule):
        """
        Generate memory occupancy intervals for specific schedule
        Schedule should only contain jobs, registered in the simulation
        :param schedule: schedule (list of jobs in execution order)
        :return: dictionary, where key (string) = name of memory, value ([SimMemoryOccupancyInterval])
        is the set of memory occupancy intervals for specific schedule
        """
        mem_access_by_mem_name = self.mem_access_grouped_by_mem_name()
        mem_occupancy_intervals_by_name = {}
        # traverse every memory record
        for item in mem_access_by_mem_name.items():
            mem_name, records = item
            ##################################
            # create occupancy intervals for memory mem_name
            mem_occupancy_intervals = []
            stored_tokens = 0
            # first occupancy interval
            cur_occupancy_interval = SimMemoryOccupancyInterval(-1, -1)

            for record in records:
                mem_access_exec_step = self.find_step_in_schedule(schedule, record.task, record.job)
                # process new interval: change start step
                if cur_occupancy_interval.start_step == -1 and cur_occupancy_interval.end_step == -1:
                    cur_occupancy_interval.start_step = mem_access_exec_step

                # change end step
                cur_occupancy_interval.end_step = max(cur_occupancy_interval.end_step, mem_access_exec_step)

                # upd number of tokens stored
                if record.action == "write":
                    stored_tokens += record.tokens
                if record.action == "read":
                    stored_tokens -= record.tokens

                cur_occupancy_interval.max_tokens = max(cur_occupancy_interval.max_tokens, stored_tokens)

                # if the last record freed the memory up, make a new interval
                if stored_tokens == 0:
                    mem_occupancy_intervals.append(cur_occupancy_interval)
                    cur_occupancy_interval = SimMemoryOccupancyInterval(-1, -1)
            # add memory occupancy intervals to the output dictionary
            mem_occupancy_intervals_by_name[mem_name] = mem_occupancy_intervals
        return mem_occupancy_intervals_by_name

    def generate_memory_occupancy_intervals_per_memory_first_in_last_out(self, schedule):
        """
        TODO: more rough estimation
        Generate memory occupancy intervals for specific schedule (first-in, last-out intervals)
        Schedule should only contain jobs, registered in the simulation
        :param schedule: schedule (list of jobs in execution order)
        :return: dictionary, where key (string) = name of memory, value ([SimMemoryOccupancyInterval])
        is the set of memory occupancy intervals for specific schedule
        """
        mem_access_by_mem_name = self.mem_access_grouped_by_mem_name()
        mem_occupancy_intervals_by_name = {}
        # traverse every memory record
        for item in mem_access_by_mem_name.items():
            mem_name, records = item
            ##################################
            # create a single occupancy interval for memory mem_name
            cur_occupancy_interval = SimMemoryOccupancyInterval(-1, -1)
            stored_tokens = 0

            for record in records:
                mem_access_exec_step = self.find_step_in_schedule(schedule, record.task, record.job)
                # process new interval: change start step
                if cur_occupancy_interval.start_step == -1 and cur_occupancy_interval.end_step == -1:
                    cur_occupancy_interval.start_step = mem_access_exec_step

                # change start step
                cur_occupancy_interval.start_step = min(cur_occupancy_interval.start_step, mem_access_exec_step)
                # change end step
                cur_occupancy_interval.end_step = max(cur_occupancy_interval.end_step, mem_access_exec_step)

                # upd number of tokens stored
                if record.action == "write":
                    stored_tokens += record.tokens
                if record.action == "read":
                    stored_tokens -= record.tokens

                cur_occupancy_interval.max_tokens = max(cur_occupancy_interval.max_tokens, stored_tokens)

            # add single memory occupancy interval to the output dictionary
            mem_occupancy_intervals_by_name[mem_name] = [cur_occupancy_interval]
        return mem_occupancy_intervals_by_name

    """
    def merge_memory_occupancy_intervals_per_memory(self, mem_occupancy_intervals_by_name):
        
        :param mem_occupancy_intervals_by_name: dictionary, where
        key (string) = name of memory,
        value ([SimMemoryOccupancyInterval]) is the set of memory occupancy intervals for specific schedule
        :return: mem_occupancy_intervals_by_name_merged: dictionary, where
        key (string) = name of memory,
        value ([SimMemoryOccupancyInterval]) is the set of memory occupancy intervals for specific schedule,
        such that all adjacent intervals are merged in one interval (e.g. intervals [3,4], [4,5], [5,6,7]
        will be merged into a single interval [3,7]

        mem_occupancy_intervals_by_name_merged = {}
        for item in mem_occupancy_intervals_by_name.items():
            mem_name, occupancy_intervals = item
            merged_intervals = []
            # if there are intervals to merge
            if len(occupancy_intervals) > 0:
                # process first interval
                first_interval = occupancy_intervals[0]
                cur_merged_interval = SimMemoryOccupancyInterval(first_interval.start_step,
                                                                 first_interval.end_step)

                for interval_id in range(1, len(occupancy_intervals)):
                    interval = occupancy_intervals[interval_id]
                    # if this interval can be merged with previous->merge
                    if interval.start_step == cur_merged_interval.end_step:
                        cur_merged_interval.end_step = interval.end_step
                    else:
                        merged_intervals.append(cur_merged_interval)
                        cur_merged_interval = SimMemoryOccupancyInterval(interval.start_step,
                                                                         interval.end_step)
                # add last interval
                if cur_merged_interval not in merged_intervals:
                    merged_intervals.append(cur_merged_interval)

            # add record
            mem_occupancy_intervals_by_name_merged[mem_name] = merged_intervals

        return mem_occupancy_intervals_by_name_merged
    """

    ##################
    # print functions

    def print_asap_schedule(self, specify_jobs=True):
        print("schedule of", len(self.jobs), "steps")
        step_id = 1
        schedule = self.get_asap_schedule()
        for job in schedule:
            step = job.task
            if specify_jobs:
                step += "(" + job.job + ")"
            print("step", step_id, ":", step)
            step_id += 1

    def print_trace(self, print_jobs=True, print_memory_access=True):
        if print_jobs:
            print("Jobs execution: ")
            for job in self.jobs:
                print(" ", job)
        if print_memory_access:
            print("Memory access: ")
            for mem_access in self.memory_accesses:
                print(" ", mem_access)

    def print_memory_access_per_edge(self):
        print("Memory access per edge")
        mem_access_by_mem_name = self.mem_access_grouped_by_mem_name()
        for item in mem_access_by_mem_name.items():
            mem_name, mem_accesses = item
            print("edge: ", mem_name)
            for mem_access in mem_accesses:
                print(" {actor: " + mem_access.task + "(" + mem_access.job + ")" +
                      ", action: " + mem_access.action + ", tokens: " + str(mem_access.tokens) +
                      "}")

    def print_memory_occupancy(self, schedule):
        mem_occupancy_per_mem_name = self.generate_memory_occupancy_intervals_per_memory_first_in_last_out(schedule)
        for item in mem_occupancy_per_mem_name.items():
            mem_name, occupancy_intervals = item
            # print memory name
            print(mem_name)
            # print occupancy intervals
            print("occupancy intervals")
            for occupancy_interval in occupancy_intervals:
                # print(" ", occupancy_interval)
                print("{start step: " + str(occupancy_interval.start_step + 1) +
                      ", end step: " + str(occupancy_interval.end_step + 1) +
                      ", max stored tokens: " + str(occupancy_interval.max_tokens) + "}")

    def mem_access_grouped_by_mem_name(self):
        """
        group memory access by memory name
        :return: dictionary, where key = memory name,
            value = list of memory access records to the memory with specified name
        """
        mem_access_by_mem_name = {}
        for mem_access in self.memory_accesses:
            if mem_access.mem_name not in mem_access_by_mem_name.keys():
                mem_access_by_mem_name[mem_access.mem_name] = []
            mem_access_by_mem_name[mem_access.mem_name].append(mem_access)
        return mem_access_by_mem_name

    def print_proc_trace(self, proc_name):
        for job in self.jobs:
            if job.processor_name == proc_name:
                print(job)

    def print_trace_per_task(self, print_jobs=False):
        tasks = self.get_tasks()
        for task in tasks:
            start_time, end_time = self.get_task_time(task)
            print("task:", task, "start:", start_time, "end:", end_time)
            if print_jobs:
                for job in self.jobs:
                    if job.task == task:
                        print("job:", job.job, ", processor:", job.processor_name,
                              "start: ", job.start_step, "end: ", job.end_time)

    def print_mem_use_per_task(self, print_time=True):
        tasks = self.get_tasks()
        for task in tasks:
            print("task:", task)
            memory_names = self.get_task_memories(task)
            for mem_name in memory_names:
                if print_time:
                    start, end = self.get_task_memory_use_time(task, mem_name)
                    print(" memory: ", mem_name, "time:", start, "to", end)
                else:
                    print(" memory: ", mem_name)

