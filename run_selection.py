def select_chromosome(ga_json_output_path):
    """
    Select a chromosome from a pareto front
    :param ga_json_output_path: path to the pareto front
    saved in the .json file
    :return: chromosome
    """
    pass


def print_pareto(ga_json_output_path):
    """
    Print pareto front
    :param ga_json_output_path: path to the pareto front
    saved in the .json file
    :return: chromosome
    """
    from converters.json_converters.json_to_ga_result import parse_json_ga_result
    chromosomes = parse_json_ga_result(ga_json_output_path)
    delay_per_phase_ms = 0.0005
    buf_size_round = 4
    # sort chromosomes by buffer size (in MB)
    chromosomes = sorted(chromosomes, key=lambda x: x.buf_size, reverse=False)
    print("BUF SIZE (MB); PHASES")
    for chromosome in chromosomes:
        buf_size = chromosome.buf_size
        phases = int(chromosome.time_loss/delay_per_phase_ms)
        print(round(buf_size, buf_size_round), ";", phases)


app1_ga_result = "/home/svetlana/Documents/mms_ga/output/app1_0_9585_9hrs.json"
app1_ga_result02 = "/home/svetlana/Documents/mms_ga/output_2/app1_4_152_50ep_2hr_02_preset.json"


app2_ga_result = "/home/svetlana/Documents/mms_ga/output/app2_6_66_201ep_21hr.json"

app3_ga_result = "/home/svetlana/Documents/mms_ga/output/app3_1_977_97ep_4hr.json"
app4_ga_result = "/home/svetlana/Documents/mms_ga/output/app4_10_69_62ep_5hr.json"
app5_ga_result = "/home/svetlana/Documents/mms_ga/output/app5_10_838_135ep_43hr.json"
app6_ga_result = "/home/svetlana/Documents/mms_ga/output/app6_9_255_164ep_21hr.json"

app_resnet_ga_result = "/home/svetlana/Documents/mms_ga/output_2/app_resnet50.json"
print_pareto(app_resnet_ga_result)
